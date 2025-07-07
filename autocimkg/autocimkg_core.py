import numpy as np
import logging
import io
import sys

from datetime import datetime
from typing import Tuple, Any
from importlib import reload
from sklearn.metrics.pairwise import cosine_similarity
from .ientities_extraction import iEntitiesExtractor
from .irelations_extraction import iRelationsExtractor
from .utils import Matcher, LLMIntegrator
from .models import KnowledgeGraph, Entity, EntityProperties, Relationship, RelationshipProperties, Document, Employee, Ontology, AlignedEntity, Log


class AutoCimKGCore:
    """
    Designed to extract knowledge from text and structure it into a knowledge graph using
    entity and relationship extraction powered by language models.
    """

    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5):
        """
        Initializes the AutoCimKGCore with specified language model, embeddings model, and operational parameters.
        
        :param llm_model: Language model instance to be used for extracting entities and relationships from text
        :param embeddings_model: Embeddings model instance to be used for creating vector representations of extracted entities
        :param sleep_time: Time to wait (in seconds) when encountering rate limits or errors (defaults to 5 seconds)
        """

        self.agent = "AutoCimKG"
        self.conf = [] # list of build_graph() conf dicts

        self.llm_model = llm_model
        self.embeddings_model = embeddings_model

        reload(logging)
        self.log = [] # list of log lists
        self.logger = logging.getLogger("autocimkg")

        self.llm_integrator = LLMIntegrator(llm_model=llm_model,
                                            embeddings_model=embeddings_model,
                                            logger=self.logger,
                                            sleep_time=sleep_time)

        self.ientities_extractor = iEntitiesExtractor(llm_model=llm_model,
                                                      embeddings_model=embeddings_model,
                                                      logger=self.logger,
                                                      sleep_time=sleep_time)

        self.irelations_extractor = iRelationsExtractor(llm_model=llm_model,
                                                        embeddings_model=embeddings_model,
                                                        logger=self.logger,
                                                        sleep_time=sleep_time)
        self.matcher = Matcher(logger=self.logger)

        self.protected_kg_resources = {
            "expert_label": "Expert",
            "department_label": "Department",
            "company_label": "Company",
            "works_in_name": "works_in",
            "part_of_name": "part_of",
            "knows_name": "knows",
            "topic_label": "Topic",
            "subsumes_name": "subsumes",
            "document_label": "Document",
            "written_by_name": "written_by"
        }

        # stdout logger
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                      datefmt="%Y-%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

    def __init_logger(self):
        """
        Initializes the logger to allow for buffering (!), besides stdout.

        :returns: String IO buffer + String IO handler
        """

        # buffered logger (i.e. imitate file)
        logging_buffer = io.StringIO()
        string_handler = logging.StreamHandler(logging_buffer)
        string_handler.setFormatter(self.formatter)
        string_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(string_handler)

        return logging_buffer, string_handler

    @staticmethod
    def __read_logging_buffer(logging_buffer: io.StringIO):
        """
        Reads and formats the buffered log.

        :param logging_buffer: String IO buffer
        """

        log = logging_buffer.getvalue()
        logs_raw = list(filter(None, log.split('\n')))

        # fmt = yyyy-mm-dd H:M:S+TZ LVL autocimkg: message
        logs = []
        for log_raw in logs_raw:
            meta = log_raw.split(" autocimkg: ")[0]
            ts_str = meta.split(" ")[0] + " " + meta.split(" ")[1]
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            log_lvl = meta.split(" ")[2]
            message = log_raw.split(" autocimkg: ")[1]
            logs.append(Log(ts=ts, logger_name="autocimkg", log_level=log_lvl, message=message))

        return logs

    def __destroy_logger(self, logging_buffer: io.StringIO, string_handler: logging.Handler):
        """
        De-initializes the logger in terms of buffering (!).

        :param logging_buffer: String IO buffer
        :param string_handler: String IO handler
        """

        self.logger.removeHandler(string_handler)
        string_handler.close()
        logging_buffer.close()

    def build_graph(self,
                    documents: list[Document],
                    employees: list[Employee],
                    max_entities_per_doc: int = 5,
                    domain: str = None,
                    ontology: Ontology = None,
                    existing_knowledge_graph: KnowledgeGraph = None,
                    expert_threshold: float = 0.7,
                    ent_threshold: float = 0.7,
                    rel_threshold: float = 0.7,
                    max_tries: int = 5,
                    max_tries_isolated_entities: int = 3,
                    entity_name_weight: float = 0.6,
                    entity_label_weight: float = 0.4
                    ) -> tuple[KnowledgeGraph, Ontology]:
        """
        Builds a knowledge graph from text by extracting entities and relationships, then integrating them into a structured graph.
        This function leverages language models to extract and merge knowledge from multiple documents of text.
        The extracted knowledge graph is enhanced with meta resources about the experts and competences (if ontology and domain are given).

        :param documents: List of Documents from which entities and relationships will be extracted, and the KG will be built
        :param employees: List of Employees which are the basis for expert determination (i.e. by using authors of the documents)
        :param max_entities_per_doc: Maximum number of entities to extract per document (defaults to 5)
        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
                                Added to the prompting of the respective LLM to pinpoint its role (defaults to None)
        :param ontology: Ontology-light to guide the knowledge graph construction. Topics get enforced or enriched (strict?).
                                Relations always get enforced (defaults to None)
        :param existing_knowledge_graph: Optional, existing knowledge graph to merge the newly extracted
                                entities and relationships into (defaults no None)
        :param expert_threshold: Threshold for matching authors and employees (i.e. determine experts).
                                A higher value indicates stricter matching (defaults to 0.7)
        :param ent_threshold: Threshold for entity matching, used to merge entities from different
                                documents. A higher value indicates stricter matching (defaults to 0.7)
        :param rel_threshold: Threshold for relationship matching, used to merge relationships from
                                different documents (defaults to 0.7)
        :param entity_name_weight: Weight of the entity name, indicating its
                                relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                                secondary significance in the evaluation process (defaults to 0.4)
        :param max_tries: Maximum number of attempts to extract entities and relationships (defaults to 5)
        :param max_tries_isolated_entities: Maximum number of attempts to process isolated entities
                                                     (i.e. entities without relationships) (defaults to 3)
        :returns: Constructed KG consisting of the merged entities and relationships extracted from text
                  + enforced or enriched ontology-light
        """

        # init log
        logging_buffer, string_handler = self.__init_logger()

        ##### RUN KGC
        self.logger.info("Performing KG construction!")
        self.logger.info("LLM chat model = %s", self.llm_model.model_dump_json())
        self.logger.info("LLM embedding model = %s", self.embeddings_model.model_dump_json())

        configuration = {"agent": self.agent, "domain": domain, "max_entities_per_doc": max_entities_per_doc,
                         "expert_threshold": expert_threshold, "ent_threshold": ent_threshold,
                         "rel_threshold": rel_threshold, "max_tries": max_tries,
                         "max_tries_isolated_entities": max_tries_isolated_entities, "entity_name_weight": entity_name_weight,
                         "entity_label_weight": entity_label_weight, "protected_resources": self.protected_kg_resources}
        self.logger.info(configuration)
        # persist construction configuration
        self.conf.append(configuration)

        ##### COMPETENCIES: EXTRACTION
        self.logger.info("Extracting entities from document %s (%s)", 1, documents[0].name)
        global_entities = self.ientities_extractor.extract_entities(context=documents[0].content,
                                                                    agent=self.agent,
                                                                    origin=documents[0].name,
                                                                    max_entities_per_doc=max_entities_per_doc,
                                                                    domain=domain,
                                                                    entity_name_weight=entity_name_weight,
                                                                    entity_label_weight=entity_label_weight)

        ##### COMPETENCIES: ONTOLOGICAL CANONICALIZATION pt1
        global_entities, id_alignments, ood_alignments = self.__align_and_partition_entities_with_ontological_topics(domain=domain,
                                                                                                                     entities=global_entities,
                                                                                                                     ontology=ontology)
        ontology_entities, ontology_relationships, ontology = self.__construct_ontological_topic_entities_and_relations(domain=domain,
                                                                                                                        ontology=ontology,
                                                                                                                        raw_entities=global_entities,
                                                                                                                        processed_entities=global_entities,
                                                                                                                        id_alignments=id_alignments,
                                                                                                                        ood_alignments=ood_alignments,
                                                                                                                        entity_name_weight=entity_name_weight,
                                                                                                                        entity_label_weight=entity_label_weight)

        ##### COMPETENCY RELATIONSHIPS: EXTRACTION
        self.logger.info("Extracting relations from document %s (%s)", 1, documents[0].name)
        global_relationships = self.irelations_extractor.extract_verify_and_correct_relations(context=documents[0].content,
                                                                                              entities=global_entities,
                                                                                              agent=self.agent,
                                                                                              origin=documents[0].name,
                                                                                              domain=domain,
                                                                                              rel_threshold=rel_threshold,
                                                                                              max_tries=max_tries,
                                                                                              max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                              entity_name_weight=entity_name_weight,
                                                                                              entity_label_weight=entity_label_weight)

        ##### COMPETENCY RELATIONSHIPS: ONTOLOGICAL CANONICALIZATION
        global_relationships = self.__align_and_filter_relationships_with_ontological_relation_types(domain=domain,
                                                                                                     relationships=global_relationships,
                                                                                                     ontology=ontology)

        ##### EXPERTS: RESOLUTION | ORG UNITS: CREATE & MAP (TO EXPERTS)
        ##### | COMPETENCY RELATIONSHIPS: MAP TO EXPERTS | DOCUMENT: CREATE & MAP (TO EXPERTS)
        expert_entities, expert_relationships = self.__construct_expert_entities_and_relations(documents[0].name,
                                                                                               documents[0].authors,
                                                                                               employees,
                                                                                               global_entities,
                                                                                               entity_name_weight,
                                                                                               entity_label_weight,
                                                                                               expert_threshold)
        global_entities.extend(expert_entities)
        global_relationships.extend(expert_relationships)

        ##### COMPETENCIES: ONTOLOGICAL CANONICALIZATION pt2
        global_entities.extend(ontology_entities)
        global_relationships.extend(ontology_relationships)

        for i in range(1, len(documents)):

            ##### COMPETENCIES: EXTRACTION pt1
            self.logger.info("Extracting entities from document %s (%s)", i + 1, documents[i].name)
            entities = self.ientities_extractor.extract_entities(context=documents[i].content,
                                                                 agent=self.agent,
                                                                 origin=documents[i].name,
                                                                 max_entities_per_doc=max_entities_per_doc,
                                                                 domain=domain,
                                                                 entity_name_weight=entity_name_weight,
                                                                 entity_label_weight=entity_label_weight)

            ##### COMPETENCIES: ONTOLOGICAL CANONICALIZATION pt1 (here, cause only aligned ones should be matched and added to the globals!)
            entities, id_alignments, ood_alignments = self.__align_and_partition_entities_with_ontological_topics(domain=domain,
                                                                                                                  entities=entities,
                                                                                                                  ontology=ontology)

            ##### COMPETENCIES: EXTRACTION pt2
            processed_entities, global_entities = self.matcher.process_lists(list1=entities, list2=global_entities, threshold=ent_threshold,
                                                                             protected=list(self.protected_kg_resources.values()),
                                                                             propagate_md=True)
            self.matcher.propagate_metadata_from_entities_to_relationships(global_entities, global_relationships)

            ##### COMPETENCIES: ONTOLOGICAL CANONICALIZATION pt2 (here, cause otherwise matched=vanished entities may leave behind topics / mappings!)
            ontology_entities, ontology_relationships, ontology = self.__construct_ontological_topic_entities_and_relations(domain=domain,
                                                                                                                            ontology=ontology,
                                                                                                                            raw_entities=entities,
                                                                                                                            processed_entities=processed_entities,
                                                                                                                            id_alignments=id_alignments,
                                                                                                                            ood_alignments=ood_alignments,
                                                                                                                            entity_name_weight=entity_name_weight,
                                                                                                                            entity_label_weight=entity_label_weight)

            global_entities.extend(ontology_entities)
            global_relationships.extend(ontology_relationships)

            ##### COMPETENCY RELATIONSHIPS: EXTRACTION pt1
            self.logger.info("Extracting relations from document %s (%s)", i + 1, documents[i].name)
            relationships = self.irelations_extractor.extract_verify_and_correct_relations(context=documents[i].content,
                                                                                           entities=processed_entities,
                                                                                           agent=self.agent,
                                                                                           origin=documents[i].name,
                                                                                           domain=domain,
                                                                                           rel_threshold=rel_threshold,
                                                                                           max_tries=max_tries,
                                                                                           max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                           entity_name_weight=entity_name_weight,
                                                                                           entity_label_weight=entity_label_weight)

            ##### COMPETENCY RELATIONSHIPS: ONTOLOGICAL CANONICALIZATION (here, cause only aligned ones should be matched and added to the globals!)
            relationships = self.__align_and_filter_relationships_with_ontological_relation_types(domain=domain,
                                                                                                  relationships=relationships,
                                                                                                  ontology=ontology)

            ##### COMPETENCY RELATIONSHIPS: EXTRACTION pt2
            processed_relationships, _ = self.matcher.process_lists(list1=relationships, list2=global_relationships, threshold=rel_threshold,
                                                                    protected=list(self.protected_kg_resources.values()))
            global_relationships.extend(processed_relationships)

            ##### EXPERTS: RESOLUTION | ORG UNITS: CREATE & MAP (TO EXPERTS)
            ##### | COMPETENCY RELATIONSHIPS: MAP TO EXPERTS | DOCUMENT: CREATE & MAP (TO EXPERTS)
            expert_entities, expert_relationships = self.__construct_expert_entities_and_relations(documents[i].name,
                                                                                                   documents[i].authors,
                                                                                                   employees,
                                                                                                   processed_entities,
                                                                                                   entity_name_weight,
                                                                                                   entity_label_weight,
                                                                                                   expert_threshold)

            global_entities.extend(expert_entities)
            global_relationships.extend(expert_relationships)

        if existing_knowledge_graph:
            self.logger.info("Matching global, EXTRACTED entities and relationships with EXISTING entities and relationships ...")
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(
                entities1=global_entities,
                entities2=existing_knowledge_graph.entities,
                relationships1=global_relationships,
                relationships2=existing_knowledge_graph.relationships,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
                protected_entities_and_relations=list(self.protected_kg_resources.values()))
            # spare special res from being replaced or being the replacement -> correctness, del duplicates later!

        constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)

        # DROP DUPLICATES (e.g. various relations, expert entities, ...) -> propagate md, drop (& propagate md again) ...
        self.matcher.propagate_metadata_in_entities(constructed_kg.entities)
        constructed_kg.remove_duplicate_entities()
        self.matcher.propagate_metadata_from_entities_to_relationships(constructed_kg.entities, constructed_kg.relationships)
        self.matcher.propagate_metadata_in_relationships(constructed_kg.relationships)
        constructed_kg.remove_duplicate_relationships()

        # create post-construction log and destroy io logger
        self.log.append(self.__read_logging_buffer(logging_buffer))
        self.__destroy_logger(logging_buffer, string_handler)

        return constructed_kg, ontology

    def __determine_expert(self, author: str, employees: list[Employee], threshold: float = 0.7):
        """
        Matches an author of a document to an employee.
        NOTE: author and employee embeddings should both be based on either lower- or upper-case names!

        :param author: Name of the author of the document
        :param employees: List of Employees which are the basis for expert determination
        :param threshold: Threshold for matching. A higher value indicates stricter
                            matching (defaults to 0.7)
        :returns: Expert (i.e. Employee) or None (if no match has been found)
        """

        author_emb = self.llm_integrator.calculate_embeddings(author).reshape(1, -1)
        cosine_similarities = [cosine_similarity(author_emb, np.array(employee.name_embedding).reshape(1, -1))[0][0] for
                               employee in employees]

        if max(cosine_similarities) > threshold:
            self.logger.info("Author '%s' matched with employee '%s' (cossim = %s > threshold = %s)",
                             author,
                             employees[cosine_similarities.index(max(cosine_similarities))].name.lower(),
                             max(cosine_similarities),
                             threshold)
            return employees[cosine_similarities.index(max(cosine_similarities))]
        else:
            self.logger.info("Author '%s' could not be matched with most similar employee '%s' (cossim = %s < threshold = %s)",
                             author,
                             employees[cosine_similarities.index(max(cosine_similarities))].name.lower(),
                             max(cosine_similarities),
                             threshold)
            return None

    def __construct_expert_entities_and_relations(self, doc_name:str, expert_candidates: list[str], employees: list[Employee],
                                                  competencies: list[Entity], entity_name_weight: float = 0.6,
                                                  entity_label_weight: float = 0.4, threshold: float = 0.7):
        """
        Constructs entities and relations about the experts linked with a document.
        E.g.: document, expert, competence, department, company and the like.

        :param doc_name: Name of the source document
        :param expert_candidates: Names of the authors of the document
        :param employees: List of Employees (i.e. base data) which are the basis for expert determination
        :param competencies: List of competencies of the authors of a document
        :param entity_name_weight: Weight of the entity name, indicating its
                            relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                            secondary significance in the evaluation process (defaults to 0.4)
        :param threshold: Threshold for matching. A higher value indicates stricter
                            matching (defaults to 0.7)
        :returns: (expert entities, expert relationships)
        """

        # nothing to construct (e.g. no competences means no meaningful, novel expert nodes or edges!)
        if expert_candidates is None or len(expert_candidates) == 0: return [], []
        if employees is None or len(employees) == 0: return [], []
        if competencies is None or len(competencies) == 0: return [], []

        self.logger.info("Constructing expert entities and relations based on authors and employees...")

        entities = []
        relationships = []
        for expert_candidate in expert_candidates:

            # resolution based on full, lowercase name
            expert = self.__determine_expert(expert_candidate.lower(), employees, threshold)
            if expert is not None:

                # expert resolution
                expert_entity = Entity(label=self.protected_kg_resources["expert_label"], name=expert.name,
                                       properties=EntityProperties(generated_at_time=datetime.now(),
                                                                   invalidated_at_time=None if expert.status == True else datetime.now(),
                                                                   agents=[self.agent],
                                                                   origins=[doc_name]))
                expert_entity.embed_entity(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                    entity_label_weight=entity_label_weight,
                    entity_name_weight=entity_name_weight
                )
                entities.append(expert_entity)
                self.logger.info("Entity (%s:%s) created",
                                 expert_entity.name, expert_entity.label)

                # document creation
                document_entity = Entity(label=self.protected_kg_resources["document_label"], name=doc_name,
                                         properties=EntityProperties(generated_at_time=datetime.now(),
                                                                     invalidated_at_time=None,
                                                                     agents=[self.agent],
                                                                     origins=[doc_name]))
                document_entity.embed_entity(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                    entity_label_weight=entity_label_weight,
                    entity_name_weight=entity_name_weight
                )
                entities.append(document_entity)
                self.logger.info("Entity (%s:%s) created",
                                 document_entity.name, document_entity.label)

                # document mapping
                document_expert_relationship = Relationship(startEntity=document_entity,
                                                            endEntity=expert_entity,
                                                            name=self.protected_kg_resources["written_by_name"],
                                                            properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                              invalidated_at_time=None,
                                                                                              agents=[self.agent],
                                                                                              origins=[doc_name]))
                document_expert_relationship.embed_relationship(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                )
                relationships.append(document_expert_relationship)
                self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                                 document_expert_relationship.startEntity.name,
                                 document_expert_relationship.startEntity.label,
                                 document_expert_relationship.name,
                                 document_expert_relationship.endEntity.name,
                                 document_expert_relationship.endEntity.label)

                # org unit creation
                department_entity = Entity(label=self.protected_kg_resources["department_label"], name=expert.department,
                                           properties=EntityProperties(generated_at_time=datetime.now(),
                                                                       invalidated_at_time=None,
                                                                       agents=[self.agent],
                                                                       origins=["employee base data"]))
                department_entity.embed_entity(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                    entity_label_weight=entity_label_weight,
                    entity_name_weight=entity_name_weight
                )
                entities.append(department_entity)
                self.logger.info("Entity (%s:%s) created",
                                 department_entity.name, department_entity.label)

                company_entity = Entity(label=self.protected_kg_resources["company_label"], name=expert.company,
                                        properties=EntityProperties(generated_at_time=datetime.now(),
                                                                    invalidated_at_time=None,
                                                                    agents=[self.agent],
                                                                    origins=["employee base data"]))
                company_entity.embed_entity(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                    entity_label_weight=entity_label_weight,
                    entity_name_weight=entity_name_weight
                )
                entities.append(company_entity)
                self.logger.info("Entity (%s:%s) created",
                                 company_entity.name, company_entity.label)

                # org unit mapping
                department_company_relationship = Relationship(startEntity=department_entity,
                                                               endEntity=company_entity,
                                                               name=self.protected_kg_resources["part_of_name"],
                                                               properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                                 invalidated_at_time=None,
                                                                                                 agents=[self.agent],
                                                                                                 origins=["employee base data"]))
                department_company_relationship.embed_relationship(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                )
                relationships.append(department_company_relationship)
                self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                                 department_company_relationship.startEntity.name,
                                 department_company_relationship.startEntity.label,
                                 department_company_relationship.name,
                                 department_company_relationship.endEntity.name,
                                 department_company_relationship.endEntity.label)

                expert_org_unit_relationship = Relationship(startEntity=expert_entity,
                                                            endEntity=department_entity,
                                                            name=self.protected_kg_resources["works_in_name"],
                                                            properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                              invalidated_at_time=None,
                                                                                              agents=[self.agent],
                                                                                              origins=["employee base data"]))
                expert_org_unit_relationship.embed_relationship(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                )
                relationships.append(expert_org_unit_relationship)
                self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                                 expert_org_unit_relationship.startEntity.name,
                                 expert_org_unit_relationship.startEntity.label,
                                 expert_org_unit_relationship.name,
                                 expert_org_unit_relationship.endEntity.name,
                                 expert_org_unit_relationship.endEntity.label)

                # competency mapping
                for competence_entity in competencies:
                    expert_competence_relationship = Relationship(startEntity= expert_entity,
                                                                  endEntity = competence_entity,
                                                                  name=self.protected_kg_resources["knows_name"],
                                                                  properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                                    invalidated_at_time=None,
                                                                                                    agents=[self.agent],
                                                                                                    origins=[doc_name]))
                    expert_competence_relationship.embed_relationship(
                        embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                    )
                    relationships.append(expert_competence_relationship)
                    self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                                     expert_competence_relationship.startEntity.name,
                                     expert_competence_relationship.startEntity.label,
                                     expert_competence_relationship.name,
                                     expert_competence_relationship.endEntity.name,
                                     expert_competence_relationship.endEntity.label)

        return entities, relationships

    def __align_and_partition_entities_with_ontological_topics(self, domain: str, entities: list[Entity], ontology: Ontology):
        """
        Aligns entities according to ontological topics.
        Strict mode in the ontology filters entities, otherwise
        out-of-distribution entities just get assigned "None".

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
        :param entities: List of entities to align
        :param ontology: Ontology w/ topics for alignment
        :return: processed entities, in-distribution alignments, out-of-distribution alignments
        """

        if entities is None or len(entities) == 0: return [], None, None

        # no alignment desired
        if ontology is None or ontology.topics is None or len(ontology.topics) == 0: return entities, None, None
        # no alignment possible
        if domain is None: return entities, None, None

        ## ALIGN
        id_entities = []  # "in-distribution"
        ood_entities = []  # "out-of-distribution"
        id_alignments = []  # name, label, topic
        ood_alignments = []  # name, label, topic

        alignments = self.ientities_extractor.align_entities(domain=domain, entities=entities, ontology=ontology)
        if alignments is None or len(alignments) == 0: return entities, None, None

        self.logger.info("Aligning and partitioning entities according to ontological topics (strict mode = %s)...",
                         ontology.strict)

        for alignment in alignments:

            # competence determination
            entity = next(
                (entity for entity in entities if alignment.name == entity.name and alignment.label == entity.label),
                None)
            if entity is None: continue  # continue if entity got invented by LLM (i.e. avoid hallucinations) ...

            # partition
            topic = next(
                (next(iter(topic.keys())) for topic in ontology.topics if next(iter(topic.keys())) == alignment.topic),
                None)
            # "None" or topic got invented by LLM = None
            if topic is not None:
                id_entities.append(entity)
                id_alignments.append(alignment)
                self.logger.info("Entity (%s:%s) = in-distribution with topic '%s'",
                                 entity.name, entity.label, alignment.topic)
            else:
                ood_entities.append(entity)
                ood_alignments.append(alignment)
                self.logger.info("Entity (%s:%s) = out-of-distribution",
                                 entity.name, entity.label)

        # return filtered and aligned entities
        if ontology.strict: return id_entities, id_alignments, ood_alignments

        # return all processed and aligned entities
        id_entities.extend(ood_entities)
        return id_entities, id_alignments, ood_alignments

    def __construct_ontological_topic_entities_and_relations(self, domain: str, ontology: Ontology, raw_entities: list[Entity],
                                                             processed_entities: list[Entity],
                                                             id_alignments: list[AlignedEntity],
                                                             ood_alignments: list[AlignedEntity],
                                                             entity_name_weight: float = 0.6,
                                                             entity_label_weight: float = 0.4):
        """
        Constructs entities and relations about the ontological topics linked to competences.
        E.g.: topic, topic-competence and the like.
        Strict mode in the ontology restricts the process, otherwise out-of-distribution
        entities are aligned and basis for construction as well as ontology enrichment.

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
        :param ontology: Ontology w/ topics. Basis for alignment and potentially enriched (w.r.t. the topics)
        :param raw_entities: List of entities (raw; basis for previous alignment)
        :param processed_entities: List of entities (processed; need to be filtered against entities, only unmatched ones count)
        :param id_alignments: List of in-distribution alignments
        :param ood_alignments: List of out-of-distribution alignments
        :param entity_name_weight: Weight of the entity name, indicating its
                relative importance in the overall evaluation process (defaults to 0.6)
        :param entity_label_weight: Weight of the entity label, reflecting its
                secondary significance in the evaluation process (defaults to 0.4)
        :return: created entities, created relations, (original or enriched) ontology
        """

        # no alignment desired
        if ontology is None or ontology.topics is None or len(
            ontology.topics) == 0 or id_alignments is None or ood_alignments is None: return [], [], ontology
        # no alignment possible
        if domain is None or raw_entities is None or len(raw_entities) == 0 or processed_entities is None or len(
            processed_entities) == 0: return [], [], ontology

        # determine relevant = unmatched entities (matched ones already got processed and enriched with
        # ontological topics in the form of entities and relations!)
        unmatched_entities = [entity for entity, processed_entity in zip(raw_entities, processed_entities) if
                              entity == processed_entity]
        if len(unmatched_entities) == 0: return [], [], ontology

        self.logger.info("Constructing ontological entities and relations according to found alignments (strict mode = %s)...",
                         ontology.strict)

        entities = []
        relations = []

        ## ID ALIGNMENT
        for alignment in id_alignments:

            # competence determination
            entity = next((entity for entity in unmatched_entities if
                           alignment.name == entity.name and alignment.label == entity.label), None)
            if entity is None: continue  # continue if entity got matched previously (!) or invented by LLM (i.e. avoid hallucinations) ...

            # topic determination or creation
            candidate_topic_entity = Entity(name=alignment.topic, label=self.protected_kg_resources["topic_label"],
                                            properties=EntityProperties(generated_at_time=datetime.now(),
                                                                        invalidated_at_time=None,
                                                                        agents=[self.agent],
                                                                        origins=["ontology"]))
            candidate_topic_entity.process()
            topic_entity = next(
                (topic for topic in entities if topic.name == candidate_topic_entity.name and topic.label == candidate_topic_entity.label), None)
            if topic_entity is None:
                topic_entity = candidate_topic_entity
                topic_entity.embed_entity(
                        embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                        entity_label_weight=entity_label_weight,
                        entity_name_weight=entity_name_weight
                    )
                entities.append(topic_entity)
                self.logger.info("Entity (%s:%s) created",
                                 topic_entity.name, topic_entity.label)

            # topic mapping
            topic_competence_relationship = Relationship(startEntity=topic_entity,
                                                         endEntity=entity,
                                                         name=self.protected_kg_resources["subsumes_name"],
                                                         properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                           invalidated_at_time=None,
                                                                                           agents=[self.agent],
                                                                                           origins=["ontology"]))
            topic_competence_relationship.embed_relationship(
                embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
            )
            relations.append(topic_competence_relationship)
            self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                             topic_competence_relationship.startEntity.name,
                             topic_competence_relationship.startEntity.label,
                             topic_competence_relationship.name,
                             topic_competence_relationship.endEntity.name,
                             topic_competence_relationship.endEntity.label)

        ## OOD ALIGNMENT
        if not ontology.strict:

            ood_entities = []
            for alignment in ood_alignments:
                entity = next((entity for entity in unmatched_entities if
                               alignment.name == entity.name and alignment.label == entity.label), None)
                if entity is None: continue  # continue if entity got matched previously (!) or invented by LLM (i.e. avoid hallucinations) ...
                ood_entities.append(entity)
            if len(ood_entities) == 0: return entities, relations, ontology

            ood_alignments = self.ientities_extractor.align_entities_with_novel_topics(domain=domain, entities=ood_entities,
                                                                                  ontology=ontology)
            if ood_alignments is None or len(ood_alignments) == 0: return entities, relations, ontology

            for alignment in ood_alignments:

                # competence determination
                entity = next((entity for entity in ood_entities if
                               alignment.name == entity.name and alignment.label == entity.label), None)
                if entity is None: continue  # continue if entity got invented by LLM (i.e. avoid hallucinations) ...

                # topic determination or creation
                candidate_topic_entity = Entity(name=alignment.topic, label=self.protected_kg_resources["topic_label"],
                                                properties=EntityProperties(generated_at_time=datetime.now(),
                                                                            invalidated_at_time=None,
                                                                            agents=[self.agent],
                                                                            origins=["ontology"]))
                candidate_topic_entity.process()
                topic_entity = next(
                    (topic for topic in entities if
                     topic.name == candidate_topic_entity.name and topic.label == candidate_topic_entity.label), None)
                if topic_entity is None:
                    topic_entity = candidate_topic_entity
                    topic_entity.embed_entity(
                        embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                        entity_label_weight=entity_label_weight,
                        entity_name_weight=entity_name_weight
                    )
                    entities.append(topic_entity)
                    self.logger.info("Entity (%s:%s) created",
                                     topic_entity.name, topic_entity.label)

                # topic mapping
                topic_competence_relationship = Relationship(startEntity=topic_entity,
                                                             endEntity=entity,
                                                             name=self.protected_kg_resources["subsumes_name"],
                                                             properties=RelationshipProperties(generated_at_time=datetime.now(),
                                                                                               invalidated_at_time=None,
                                                                                               agents=[self.agent],
                                                                                               origins=["ontology"]))
                topic_competence_relationship.embed_relationship(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                )
                relations.append(topic_competence_relationship)
                self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) created",
                                 topic_competence_relationship.startEntity.name,
                                 topic_competence_relationship.startEntity.label,
                                 topic_competence_relationship.name,
                                 topic_competence_relationship.endEntity.name,
                                 topic_competence_relationship.endEntity.label)

            ## ONTOLOGY ENRICHMENT (just use all unique ood alignment topics)
            novel_ontological_topics = []
            for alignment in ood_alignments: novel_ontological_topics.append(alignment.topic)
            novel_ontological_topics = list(set(novel_ontological_topics))
            novel_ontological_topics = self.ientities_extractor.enrich_ontology(domain=domain, topics=novel_ontological_topics)
            if novel_ontological_topics is None or len(
                novel_ontological_topics) == 0: return entities, relations, ontology
            for topic in novel_ontological_topics:

                topic_entity = Entity(name=topic.topic, label=self.protected_kg_resources["topic_label"])
                topic_entity.process()
                topic_entity = next((topic for topic in entities if topic.name == topic_entity.name and topic.label == topic_entity.label), None)
                if topic_entity is None: continue  # continue if topic got invented by LLM (i.e. avoid hallucinations) ...

                examples = ', '.join(topic.examples[:-1]) + ' or ' + topic.examples[-1] \
                            if isinstance(topic.examples, list) and len(topic.examples) > 0 else topic.examples
                ontology.topics.append({topic.topic: f"{topic.description.rstrip('.')} (e.g., {examples})"})
                self.logger.info("Ontological topic '%s: %s (e.g., %s)' created",
                                 topic.topic, topic.description.rstrip('.'), examples)

        return entities, relations, ontology

    def __align_and_filter_relationships_with_ontological_relation_types(self, domain: str,
                                                                         relationships: list[Relationship],
                                                                         ontology: Ontology):
        """
        Aligns relationships according to ontological relation types.
        In-distribution relationships get semantically generalized to more timeless relation types,
        out-of-distribution relationships just get filtered out.

        :param domain: Domain that encloses the knowledge graph construction (e.g. 'financial supervisory domain').
        :param relationships: List of relationships to align
        :param ontology: Ontology w/ relations for alignment
        :return: processed relationships
        """

        if relationships is None or len(relationships) == 0: return []

        # no alignment desired
        if ontology is None or ontology.relations is None or len(ontology.relations) == 0: return relationships
        # no alignment possible
        if domain is None: return relationships

        # ALIGN
        id_relationships = [] # "in-distribution"

        alignments = self.irelations_extractor.align_relationships(domain=domain,
                                                                   relationships=relationships,
                                                                   ontology=ontology)
        if alignments is None or len(alignments) == 0: return relationships

        self.logger.info("Aligning and filtering relations according to ontological relationship types...")

        for alignment in alignments:

            # relationship determination
            relationship = None
            for idx, relation in enumerate(relationships):
                if alignment.list_number == (idx + 1) \
                    and alignment.start_entity == relation.startEntity.name \
                    and alignment.end_entity == relation.endEntity.name:
                    relationship = relation
                    break
            if relationship is None: continue # continue if relationship got invented by LLM (i.e. avoid hallucinations) ...

            # filter and update
            relational_type = next((next(iter(rel.keys())) for rel in ontology.relations if next(iter(rel.keys())) == alignment.name), None)
            if relational_type is None: continue # continue if 'None' or type got invented by LLM (i.e. avoid hallucinations) ...

            self.logger.info("Relation (%s:%s) -[%s]-> (%s:%s) = in-distribution with type [%s]",
                             relationship.startEntity.name,
                             relationship.startEntity.label,
                             relationship.name,
                             relationship.endEntity.name,
                             relationship.endEntity.label,
                             relational_type)

            relationship.name = relational_type
            relationship.embed_relationship(
                    embeddings_function=lambda x: self.llm_integrator.calculate_embeddings(x),
                )
            id_relationships.append(relationship)

        # return filtered and aligned relationships
        return id_relationships

