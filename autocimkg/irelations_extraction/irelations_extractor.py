from ..utils import LLMIntegrator, RelationshipsExtractor, RelationshipAlignmentExtractor, Matcher
from ..models import Ontology, Entity, EntityProperties, Relationship, RelationshipProperties, AlignedRelationship, KnowledgeGraph
from datetime import datetime

class iRelationsExtractor:
    """
    Designed to extract relationships between entities.
    """

    def __init__(self, llm_model, embeddings_model, logger, sleep_time: int = 5):
        """
        Initializes the iRelationsExtractor with specified language model, embeddings model, and operational parameters.
        
        :param llm_model: Language model instance used for extracting relationships between entities
        :param embeddings_model: Embeddings model instance used for generating vector representations of entities and relationships
        :param logger: Logger instance used for logging operations
        :param sleep_time (int): Time to wait (in seconds) when encountering rate limits or errors (defaults to 5 seconds)
        """

        self.llm_integrator = LLMIntegrator(llm_model=llm_model,
                                             embeddings_model=embeddings_model,
                                             logger=logger,
                                             sleep_time=sleep_time)
        self.matcher = Matcher(logger=logger)
        self.logger = logger

    def __extract_relations(self,
                            context: str,
                            entities: list[Entity],
                            agent: str,
                            origin: str,
                            domain: str = None,
                            isolated_entities_without_relations: list[Entity] = None,
                            max_tries: int = 5,
                            entity_name_weight: float = 0.6,
                            entity_label_weight: float = 0.4,
                            ) -> list[Relationship]:
        """
        Extracts relationships from a given context for specified entities and add embeddings. This method handles invented entities.
        
        :param context: Textual context from which relationships will be extracted
        :param entities: List of Entity instances to be considered in the extraction
        :param agent: Agent to be set in the relation metadata
        :param origin: Origin to be set in the relation metadata
        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
                                    Added to the prompting of the respective LLM to pinpoint its role (defaults to None)
        :param isolated_entities_without_relations: List of entities without
                existing relationships to include in the extraction (defaults to None)
        :param max_tries: Maximum number of attempts to extract relationships (defaults to 5)
        :param entity_name_weight: Weight of the entity name, indicating its
                 relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                 secondary significance in the evaluation process (defaults to 0.4)
        :returns: List of extracted Relationship instances with embeddings
        """

        domain_instruction = "." if domain is None else f" in the {domain}."
        instructions = "# DIRECTIVES\n" \
                    + f"- Act like an experienced knowledge graph builder{domain_instruction}\n" \
                     + "- Extract relationships between the provided entities based on the context.\n" \
                     + "- Provided entities are represented as an array in the form (name, label).\n" \
                     + "- Adhere completely to the provided entities list.\n" \
                     + "- Do not change the name or label of the provided entities list.\n" \
                     + "- Do not add any entity outside the provided list.\n" \
                     + "- Avoid reflexive relations."

        # do not give the LLM complex data structures as context to avoid hallucinations as much as possible
        entities_simplified = [(entity.name, entity.label) for entity in entities]

        formatted_context = "# CONTEXT\n" \
                          + f"{context}\n\n" \
                           + "# ENTITIES\n" \
                          + f"{entities_simplified}"

        if isolated_entities_without_relations:
            isolated_entities_without_relations_simplified = [(entity.name, entity.label) for entity in isolated_entities_without_relations]

            instructions = "# DIRECTIVES\n" \
                        + f"- Act like an experienced knowledge graph builder{domain_instruction}\n" \
                        + f"- Based on the provided context, link the entities: \n {isolated_entities_without_relations_simplified}" \
                       + f"\n to the following entities: \n {entities_simplified}.\n" \
                         + "- Given entities are represented as an array in the form (name, label).\n" \
                         + "- Avoid reflexive relations."

            formatted_context = "# CONTEXT\n" \
                             + f"{context}"

        tries = 0
        relationships = None
        curated_relationships:list[Relationship] = []
        
        while tries < max_tries:
            try:
                relationships = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=RelationshipsExtractor
                )

                if relationships and "relationships" in relationships.keys():
                    break
                
            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1
    
        if not relationships or "relationships" not in relationships:
            self.logger.warning("Failed to extract relationships after multiple attempts")
            return []

        relationships_simplified = [(relationship["startNode"]["name"], relationship["startNode"]["label"],
                                     relationship["name"],
                                     relationship["endNode"]["name"], relationship["endNode"]["label"])
                                    for relationship in relationships["relationships"]]
        self.logger.info("Relations extracted = %s", relationships_simplified)
        
        # -------- Verification of invented entities and matching to the closest ones from the input entities-------- #
        self.logger.info("Verification of invented entities ...")
        kg_llm_output = KnowledgeGraph(relationships=[], entities=entities)
        for relationship in relationships["relationships"]:
            self.logger.info("Investigating (%s, %s) --[%s]-> (%s, %s)",
                             relationship["startNode"]["name"], relationship["startNode"]["label"],
                             relationship["name"],
                             relationship["endNode"]["name"], relationship["endNode"]["label"])

            start_entity = Entity(label=relationship["startNode"]["label"], name = relationship["startNode"]["name"],
                                  properties=EntityProperties(generated_at_time=datetime.now(),
                                                              invalidated_at_time=None,
                                                              agents=[agent],
                                                              origins=[origin]))
            end_entity = Entity(label=relationship["endNode"]["label"], name = relationship["endNode"]["name"],
                                properties=EntityProperties(generated_at_time=datetime.now(),
                                                            invalidated_at_time=None,
                                                            agents=[agent],
                                                            origins=[origin]))
            
            start_entity.process()
            end_entity.process()
            
            start_entity_in_input_entities = kg_llm_output.get_entity(start_entity)
            end_entity_in_input_entities = kg_llm_output.get_entity(end_entity)
            
            if start_entity_in_input_entities is not None and end_entity_in_input_entities is not None :
                curated_relationships.append(Relationship(startEntity= start_entity_in_input_entities,
                                      endEntity = end_entity_in_input_entities,
                                      name = relationship["name"],
                                      properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                        invalidated_at_time=None,
                                                                        agents=[agent],
                                                                        origins=[origin])))
                
            elif start_entity_in_input_entities is None and end_entity_in_input_entities is None:
                self.logger.info("The entities (%s, %s) and (%s, %s) are INVENTED. Solving them ...",
                                 start_entity.name, start_entity.label, end_entity.name, end_entity.label)
                start_entity.embed_entity(embeddings_function=self.llm_integrator.calculate_embeddings,
                                         entity_label_weight=entity_label_weight,
                                         entity_name_weight=entity_name_weight)
                end_entity.embed_entity(embeddings_function=self.llm_integrator.calculate_embeddings,
                                       entity_label_weight=entity_label_weight,
                                       entity_name_weight=entity_name_weight)
                
                start_entity = self.matcher.find_match(obj1=start_entity, list_objects=entities, threshold=0.5, merge_md=False)
                end_entity = self.matcher.find_match(obj1=end_entity, list_objects=entities, threshold=0.5, merge_md=False)
                
                curated_relationships.append(Relationship(startEntity= start_entity,
                                      endEntity = end_entity,
                                      name = relationship["name"],
                                      properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                        invalidated_at_time=None,
                                                                        agents=[agent],
                                                                        origins=[origin])))
                
            elif start_entity_in_input_entities is None:
                self.logger.info("The entity (%s, %s) is INVENTED. Solving it ...",
                                 start_entity.name, start_entity.label)
                start_entity.embed_entity(embeddings_function=self.llm_integrator.calculate_embeddings,
                                         entity_label_weight=entity_label_weight,
                                         entity_name_weight=entity_name_weight)
                start_entity = self.matcher.find_match(obj1=start_entity, list_objects=entities, threshold=0.5, merge_md=False)
                
                curated_relationships.append(Relationship(startEntity= start_entity,
                                      endEntity = end_entity,
                                      name = relationship["name"],
                                      properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                        invalidated_at_time=None,
                                                                        agents=[agent],
                                                                        origins=[origin])))
                
            elif end_entity_in_input_entities is None:
                self.logger.info("The entity (%s, %s) is INVENTED. Solving it ...",
                                 end_entity.name, end_entity.label)
                end_entity.embed_entity(embeddings_function=self.llm_integrator.calculate_embeddings,
                                       entity_label_weight=entity_label_weight,
                                       entity_name_weight=entity_name_weight)
                end_entity = self.matcher.find_match(obj1=end_entity, list_objects=entities, threshold=0.5, merge_md=False)
                
                curated_relationships.append(Relationship(startEntity= start_entity,
                                      endEntity = end_entity,
                                      name = relationship["name"],
                                      properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                        invalidated_at_time=None,
                                                                        agents=[agent],
                                                                        origins=[origin])))
        
        kg = KnowledgeGraph(relationships = curated_relationships, entities=entities)
        kg.embed_relationships(
            embeddings_function=lambda x:self.llm_integrator.calculate_embeddings(x)
            )
        return kg.relationships
    
    
    def extract_verify_and_correct_relations(self,
                          context: str, 
                          entities: list[Entity],
                          agent: str,
                          origin: str,
                          domain: str = None,
                          rel_threshold: float = 0.7,
                          max_tries: int = 5,
                          max_tries_isolated_entities: int = 3,
                          entity_name_weight: float = 0.6,
                          entity_label_weight: float = 0.4) -> list[Relationship]:
        """
        Extract, verify, and correct relationships between entities in the given context.

        :param context: Textual context for extracting relationships
        :param entities: List of Entity instances to consider
        :param agent: Agent to be set in the relation metadata
        :param origin: Origin to be set in the relation metadata
        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
                                    Added to the prompting of the respective LLM to pinpoint its role (defaults to None)
        :param rel_threshold: Threshold for matching corrected relationships (defaults to 0.7)
        :param max_tries: Maximum number of attempts to extract relationships (defaults to 5)
        :param max_tries_isolated_entities: Maximum number of attempts to process isolated entities (defaults to 3)
        :param entity_name_weight: Weight of the entity name, indicating its
                                     relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                                      secondary significance in the evaluation process (defaults to 0.4)
        :returns: List of curated Relationship instances after verification and correction
        """

        # nothing to extract
        if entities is None or len(entities) == 0: return []

        tries = 0
        isolated_entities_without_relations:list[Entity]= []
        curated_relationships = self.__extract_relations(context=context,
                                                         entities=entities,
                                                         agent=agent,
                                                         origin=origin,
                                                         domain=domain,
                                                         max_tries=max_tries,
                                                         entity_name_weight=entity_name_weight,
                                                         entity_label_weight=entity_label_weight)
        
        # -------- Verification of isolated entities without relations and re-prompting the LLM accordingly-------- #   
        isolated_entities_without_relations = KnowledgeGraph(entities=entities, 
                                                             relationships=curated_relationships).find_isolated_entities()
        
        while tries < max_tries_isolated_entities and isolated_entities_without_relations:
            isolated_entities_simplified = [(entity.name, entity.label) for entity in isolated_entities_without_relations]
            self.logger.info("Isolated entities without relations = %s. Solving them ... (Attempt %s/%s)", isolated_entities_simplified, tries+1, max_tries_isolated_entities)

            corrected_relationships = self.__extract_relations(context = context,
                                                               entities=isolated_entities_without_relations,
                                                               agent=agent,
                                                               origin=origin,
                                                               domain=domain,
                                                               isolated_entities_without_relations=isolated_entities_without_relations,
                                                               entity_name_weight=entity_name_weight,
                                                               entity_label_weight=entity_label_weight)
            matched_corrected_relationships, _ = self.matcher.process_lists(list1 = corrected_relationships, list2=curated_relationships, threshold=rel_threshold)
            curated_relationships.extend(matched_corrected_relationships)
                
            isolated_entities_without_relations = KnowledgeGraph(entities=entities, relationships=corrected_relationships).find_isolated_entities()
            tries += 1
        return curated_relationships

    def align_relationships(self, domain: str, relationships: list[Relationship], ontology: Ontology, max_tries: int = 5) \
        -> list[AlignedRelationship]:
        """
        Aligns relationships based on the provided ontological relations.
        'None' indicates that no alignment is possible within the given ontology.

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain')
        :param relationships: List of relationships to be aligned
        :param ontology: Ontology that will be used to align relationships
        :param max_tries: Maximum number of attempts to align relationships (defaults to 5)
        :returns: List of aligned relationships
        """

        instructions = "# DIRECTIVES\n" \
                    + f"- Act like an experienced knowledge graph builder in the {domain}.\n" \
                     + "- Match the provided relationships with given types based on the ontology.\n" \
                     + "- The ontology lists general relationship types and respective descriptions in the context of the domain.\n" \
                     + "- Provided relationships are represented as a numbered list of relational triplets in the form " \
                       + "(start_entity, RELATIONSHIP, end_entity).\n" \
                     + "- For each relational triple, first think about its meaning in the domain at hand and then pick the " \
                       + "relationship type from the ontology that is the most appropriate to characterize and therefore " \
                       + "replace the relationship between the respective entities.\n" \
                     + "- If you think that no relationship type is suitable to reflect the essence of a " \
                       + "respective relationship, pick 'None'."

        # do not give the LLM complex data structures as context to avoid hallucinations as much as possible
        ontology_simplified = ""
        index = 1
        for relation in ontology.relations:
            # fetch only key/value pair per topic...
            ontology_simplified += f"{index}) {next(iter(relation.keys()))}: {next(iter(relation.values()))}\n"
            index += 1
        ontology_simplified += f"{index}) None: No listed relationship type is an appropriate replacement.\n"

        relationships_simplified = ""
        for idx, relationship in enumerate(relationships):
            relationships_simplified += f"{idx+1}) ('{relationship.startEntity.name}', '{relationship.name}', '{relationship.endEntity.name}')\n"

        formatted_context = "# ONTOLOGY\n" \
                         + f"{ontology_simplified}\n" \
                          + "# RELATIONSHIPS\n" \
                         + f"{relationships_simplified}"

        tries = 0
        aligned_relationships = None

        while tries < max_tries:
            try:
                aligned_relationships = self.llm_integrator.extract_information(
                    instructions=instructions,
                    context=formatted_context,
                    output_data_structure=RelationshipAlignmentExtractor
                )

                if aligned_relationships and "relationships" in aligned_relationships.keys():
                    break

            except Exception as e:
                self.logger.exception("Error occurred. Retrying (%s/%s) ...", tries + 1, max_tries)

            tries += 1

        if not aligned_relationships or "relationships" not in aligned_relationships:
            self.logger.warning("Failed to align relationships after multiple attempts")
            return []

        aligned_relationships = [AlignedRelationship(list_number=aligned_relationship["list_number"],
                                                     start_entity=aligned_relationship["start_entity"],
                                                     name=aligned_relationship["relationship_type"],
                                                     end_entity=aligned_relationship["end_entity"]) \
                                 for aligned_relationship in aligned_relationships["relationships"]]

        return aligned_relationships