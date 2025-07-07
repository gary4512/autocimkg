from ..utils import LLMIntegrator, EntitiesExtractor, EntityAlignmentExtractor, NovelEntityAlignmentExtractor, NovelTopicExtractor
from ..models import Ontology, AlignedEntity, NovelTopic, Entity, EntityProperties, KnowledgeGraph
from datetime import datetime

class iEntitiesExtractor:
    """
    Designed to extract entities from text using natural language processing tools and embeddings.
    """

    def __init__(self, llm_model, embeddings_model, logger, sleep_time: int = 5):
        """
        Initializes the iEntitiesExtractor with specified language model, embeddings model, and operational parameters.
        
        :param llm_model: Language model instance to be used for extracting entities from text
        :param embeddings_model: Embeddings model instance to be used for generating vector representations of text entities
        :param logger: Logger instance used for logging operations
        :param sleep_time: Time to wait (in seconds) when encountering rate limits or errors (defaults to 5 seconds)
        """
    
        self.llm_integrator =  LLMIntegrator(llm_model=llm_model,
                                             embeddings_model=embeddings_model,
                                             logger=logger,
                                             sleep_time=sleep_time)
        self.logger = logger

    def extract_entities(self, context: str,
                         agent: str,
                         origin: str,
                         domain: str = None,
                         max_entities_per_doc: int = 5,
                         max_tries: int = 5,
                         entity_name_weight: float = 0.6,
                         entity_label_weight: float = 0.4) -> list[Entity]:
        """
        Extracts entities from a given context.

        :param context: Textual context from which entities will be extracted
        :param agent: Agent to be set in the entity metadata
        :param origin: Origin to be set in the entity metadata
        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
                                    Added to the prompting of the respective LLM to pinpoint its role (defaults to None)
        :param max_entities_per_doc: Maximum number of entities to extract per document (defaults to 5)
        :param max_tries: Maximum number of attempts to extract entities (defaults to 5)
        :param entity_name_weight: Weight of the entity name, indicating its
                                     relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                                      secondary significance in the evaluation process (defaults to 0.4)
        :returns: List of extracted entities with embeddings.
        """

        domain_instruction = "." if domain is None else f" in the {domain}."
        instructions  = "# DIRECTIVES\n" \
                     + f"- Act like an experienced knowledge graph builder{domain_instruction}\n" \
                      + "- Try to understand the context and extract entities that reflect its central content.\n" \
                      + "- The extracted entities should therefore characterize the knowledge (e.g. about the specialist" \
                        + " topics covered) or the skills (e.g. regarding the methods used) of the author of the context.\n" \
                      + "- Abbreviations should be spelled-out and then followed by the abbreviation in parentheses, " \
                        + "e.g. 'GPT' should be represented as 'General Pretrained Transformer (GPT)'.\n" \
                    + f"- A maximum of {max_entities_per_doc} entities should be extracted, although there may be less. Pick the most " \
                        + "significant in alignment with the instructions given."

        formatted_context = "# CONTEXT\n" \
                            + f"{context}"

        tries = 0
        entities = None

        while tries < max_tries:
            try:
                entities = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=EntitiesExtractor
                )

                if entities and "entities" in entities.keys():
                    break

            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1
    
        if not entities or "entities" not in entities:
            self.logger.warning("Failed to extract entities after multiple attempts")
            return []

        entities = [Entity(label=entity["label"], name = entity["name"],
                           properties=EntityProperties(generated_at_time=datetime.now(),
                                                       invalidated_at_time=None,
                                                       agents=[agent],
                                                       origins=[origin]))
                    for entity in entities["entities"]]

        entities_simplified = [(entity.name, entity.label) for entity in entities]
        self.logger.info("Entities extracted = %s", entities_simplified)

        kg = KnowledgeGraph(entities = entities, relationships=[])
        kg.embed_entities(
            embeddings_function=lambda x:self.llm_integrator.calculate_embeddings(x),
            entity_label_weight=entity_label_weight,
            entity_name_weight=entity_name_weight
            )
        return kg.entities

    def align_entities(self, domain: str, entities: list[Entity], ontology: Ontology, max_tries: int = 5) \
            -> list[AlignedEntity]:
        """
        Aligns entities based on the provided ontological topics.
        'None' indicates that no alignment is possible within the given ontology.

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain')
        :param entities: List of entities to be aligned
        :param ontology: Ontology that will be used to align entities
        :param max_tries: Maximum number of attempts to align entities (defaults to 5)
        :returns: List of aligned entities
        """

        instructions = "# DIRECTIVES\n" \
                    + f"- Act like an experienced knowledge graph builder in the {domain}.\n" \
                     + "- Match the provided entities with given topics based on the ontology.\n" \
                     + "- The ontology lists general topics and respective descriptions in the context of the domain.\n" \
                     + "- Provided entities are represented as an array in the form (name, label).\n" \
                     + "- For each entity, first think about its meaning in the domain at hand and then pick the " \
                       + "topic from the ontology that is the most appropriate to subsume it and therefore " \
                       + "characterize it in a broader sense.\n" \
                     + "- If you think that no topic characterizes a respective entity properly, pick 'None'."

        # do not give the LLM complex data structures as context to avoid hallucinations as much as possible
        ontology_simplified = ""
        index = 1
        for topic in ontology.topics:
            # fetch only key/value pair per topic...
            ontology_simplified += f"{index}) {next(iter(topic.keys()))}: {next(iter(topic.values()))}\n"
            index += 1
        ontology_simplified += f"{index}) None: No listed topic is an appropriate parent category for the entity.\n"
        entities_simplified = [(entity.name, entity.label) for entity in entities]
        formatted_context = "# ONTOLOGY\n" \
                         + f"{ontology_simplified}\n" \
                          + "# ENTITIES\n" \
                         + f"{entities_simplified}"

        tries = 0
        aligned_entities = None

        while tries < max_tries:
            try:
                aligned_entities = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=EntityAlignmentExtractor
                )

                if aligned_entities and "entities" in aligned_entities.keys():
                    break

            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1

        if not aligned_entities or "entities" not in aligned_entities:
            self.logger.warning("Failed to align entities after multiple attempts")
            return []

        aligned_entities = [AlignedEntity(name=aligned_entity["name"], label=aligned_entity["label"], topic=aligned_entity["topic"])
                    for aligned_entity in aligned_entities["entities"]]

        return aligned_entities

    def align_entities_with_novel_topics(self, domain: str, entities: list[Entity], ontology: Ontology, max_tries: int = 5) \
        -> list[AlignedEntity]:
        """
        Aligns entities with newly created topics (based on the provided ontological topics).

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
        :param entities: List of entities to be aligned.
        :param ontology: Ontology that will be used for reference (i.e. spare existing topics and match granularity).
        :param max_tries: Maximum number of attempts to align entities (defaults to 5)
        :returns: List of aligned entities w/ new topics.
        """

        instructions = "# DIRECTIVES\n" \
                    + f"- Act like an experienced knowledge graph builder in the {domain}.\n" \
                     + "- Propose new parent topics for the provided entities based on the given ontology.\n" \
                     + "- The already existing ontology lists general topics and respective descriptions in the " \
                       + "context of the domain.\n" \
                     + "- Provided entities are represented as an array in the form (name, label).\n" \
                     + "- For each entity, first think about its meaning in the domain at hand and then propose " \
                       + "a new topic that is the most appropriate to subsume it and therefore characterize " \
                       + "it in a broader sense.\n" \
                     + "- Take the ontology as a hint to only propose strictly new topics and match the " \
                        + "the granularity level of the existing topics."

        # do not give the LLM complex data structures as context to avoid hallucinations as much as possible
        ontology_simplified = ""
        for index, topic in enumerate(ontology.topics):
            # fetch only key/value pair per topic...
            ontology_simplified += f"{index + 1}) {next(iter(topic.keys()))}: {next(iter(topic.values()))}\n"
        entities_simplified = [(entity.name, entity.label) for entity in entities]
        formatted_context = "# ONTOLOGY\n" \
                         + f"{ontology_simplified}\n" \
                          + "# ENTITIES\n" \
                         + f"{entities_simplified}"

        tries = 0
        aligned_entities = None

        while tries < max_tries:
            try:
                aligned_entities = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=NovelEntityAlignmentExtractor
                )

                if aligned_entities and "entities" in aligned_entities.keys():
                    break

            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1

        if not aligned_entities or "entities" not in aligned_entities:
            self.logger.warning("Failed to align entities after multiple attempts")
            return []

        aligned_entities = [AlignedEntity(name=aligned_entity["name"], label=aligned_entity["label"], topic=aligned_entity["topic"])
                    for aligned_entity in aligned_entities["entities"]]

        return aligned_entities

    def enrich_ontology(self, domain: str, topics: list[str], max_tries: int = 5) -> list[NovelTopic]:
        """
        Describes novel ontological topics and enriches them with examples.

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
        :param topics: List of topics to be described and enriched.
        :param max_tries: Maximum number of attempts to align entities (defaults to 5)
        :returns: List of described and enriched topics.
        """

        instructions = "# DIRECTIVES\n" \
                    + f"- Act like an experienced knowledge graph builder in the {domain}.\n" \
                     + "- Propose concise, one sentence descriptions for the provided ontological topics in the given domain.\n" \
                     + "- Think of three specific and short examples for each topic (i.e. at best only one keyword per example)."
        topics_list = ""
        for index, topic in enumerate(topics):
            topics_list += f"{index + 1}) {topic}\n"
        formatted_context = "# TOPICS\n" \
                            + f"{topics_list}"

        tries = 0
        novel_topics = None

        while tries < max_tries:
            try:
                novel_topics = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=NovelTopicExtractor
                )

                if novel_topics and "topics" in novel_topics.keys():
                    break

            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1

        if not novel_topics or "topics" not in novel_topics:
            self.logger.warning("Failed to describe topics after multiple attempts")
            return []

        novel_topics = [
            NovelTopic(topic=novel_topic["topic"], description=novel_topic["description"], examples=novel_topic["examples"])
            for novel_topic in novel_topics["topics"]]

        return novel_topics