import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Union
from datetime import datetime

from ..models import Entity, Relationship


class Matcher:
    """
    Designed to handle the matching and processing of entities or relations based on cosine similarity or name matching.
    """

    def __init__(self, logger):
        """
        Initializes the matcher.

        :param logger: Logger instance used for logging operations
        """

        self.logger = logger

    def find_match(self, obj1: Union[Entity, Relationship], list_objects: list[Union[Entity, Relationship]],
                   threshold: float = 0.8, protected: list[str] = None, merge_md=True) -> Union[Entity, Relationship]:
        """
        Finds a matching Entity or Relationship object based on name or high cosine similarity.

        :param obj1: The Entity or Relationship to find matches for
        :param list_objects: List of Entities or Relationships to match against
        :param threshold: Cosine similarity threshold (defaults to 0.8)
        :param protected: Entity types and relationships to spare from being replaced or
            being the replacement; especially useful for special objects like experts, topics etc. (defaults to None)
        :param merge_md: Merge metadata of matching entities (defaults to True)
        :returns: Best match or the original object if no match is found
        """

        # if obj1 is protected -> return
        if protected is not None:
            if isinstance(obj1, Entity):
                if obj1.label in protected:
                    return obj1
            elif isinstance(obj1, Relationship):
                if obj1.name in protected:
                    return obj1

        # use all list_objects or filter possible match options
        list_obj = list_objects
        if protected is not None:
            list_obj = []
            for obj in list_objects:
                if isinstance(obj, Entity):
                    if obj.label not in protected:
                        list_obj.append(obj)
                elif isinstance(obj, Relationship):
                    if obj.name not in protected:
                        list_obj.append(obj)

        name1 = obj1.name
        label1 = obj1.label if isinstance(obj1, Entity) else None
        emb1 = np.array(obj1.properties.embeddings).reshape(1, -1)
        best_match = None
        best_cosine_sim = threshold

        for obj2 in list_obj:
            name2 = obj2.name
            label2 = obj2.label if isinstance(obj2, Entity) else None
            emb2 = np.array(obj2.properties.embeddings).reshape(1, -1)

            if name1 == name2 and label1 == label2:
                return obj1
            cosine_sim = cosine_similarity(emb1, emb2)[0][0]
            if cosine_sim > best_cosine_sim:
                best_cosine_sim = cosine_sim
                best_match = obj2

        if best_match:
            if isinstance(obj1, Relationship):
                self.logger.info("Relation was matched: [%s] -> [%s]", obj1.name, best_match.name)
                obj1.name = best_match.name
                obj1.properties.embeddings = best_match.properties.embeddings

            elif isinstance(obj1, Entity):
                self.logger.info("Entity was matched: (%s:%s) -> (%s:%s)", obj1.name, obj1.label, best_match.name, best_match.label)

                # merge entity metadata ...
                if merge_md:
                    self.align_resource_metadata(best_match, obj1)

                return best_match

        return obj1

    @staticmethod
    def align_resource_metadata(resource_to_update: Union[Entity, Relationship], resource_fresh: Union[Entity, Relationship]):
        """
        Analyses metadata of a base resource and updates it with new metadata, if needed.

        :param resource_to_update: Base resource
        :param resource_fresh: New resource
        """

        # generated_at_time
        if resource_fresh.properties.generated_at_time is not None and resource_to_update.properties.generated_at_time is not None:
            if resource_fresh.properties.generated_at_time > resource_to_update.properties.generated_at_time:
                resource_to_update.properties.generated_at_time = resource_fresh.properties.generated_at_time
        elif resource_fresh.properties.generated_at_time is not None and resource_to_update.properties.generated_at_time is None:
            resource_to_update.properties.generated_at_time = resource_fresh.properties.generated_at_time
        # invalidated_at_time
        if resource_fresh.properties.invalidated_at_time is not None and resource_to_update.properties.invalidated_at_time is not None:
            if resource_fresh.properties.invalidated_at_time > resource_to_update.properties.invalidated_at_time:
                resource_to_update.properties.invalidated_at_time = resource_fresh.properties.invalidated_at_time
        elif resource_fresh.properties.invalidated_at_time is not None and resource_to_update.properties.invalidated_at_time is None:
            resource_to_update.properties.invalidated_at_time = resource_fresh.properties.invalidated_at_time
        # agents
        if resource_fresh.properties.agents is not None and resource_to_update.properties.agents is not None:
            resource_to_update.properties.agents = list(set(resource_to_update.properties.agents).union(resource_fresh.properties.agents))
        elif resource_fresh.properties.agents is not None and resource_to_update.properties.agents is None:
            resource_to_update.properties.agents = resource_fresh.properties.agents
        # origins
        if resource_fresh.properties.origins is not None and resource_to_update.properties.origins is not None:
            resource_to_update.properties.origins = list(set(resource_to_update.properties.origins).union(resource_fresh.properties.origins))
        elif resource_fresh.properties.origins is not None and resource_to_update.properties.origins is None:
            resource_to_update.properties.origins = resource_fresh.properties.origins

    @staticmethod
    def propagate_metadata_from_entities_to_relationships(entities: list[Entity], relationships: list[Relationship]):
        """
        Propagates entity metadata to a list of relationships. Assumes consistent entities!
        Needed, as matching also alters global entities, which are already present in relationships -> update!

        :param entities: List of entities
        :param relationships: List of relationships
        """

        if entities is None or relationships is None: return

        for entity in entities:
            equivalent_entities = [rel.startEntity for rel in relationships if entity == rel.startEntity]
            equivalent_entities.extend([rel.endEntity for rel in relationships if entity == rel.endEntity])

            # overwrite all "update candidates" ...
            for entity_to_update in equivalent_entities:
                entity_to_update.properties.generated_at_time = entity.properties.generated_at_time
                entity_to_update.properties.invalidated_at_time = entity.properties.invalidated_at_time
                entity_to_update.properties.agents = entity.properties.agents
                entity_to_update.properties.origins = entity.properties.origins

    def propagate_metadata_in_relationships(self, list_rel: list[Relationship]):
        """
        Propagates relationship metadata in a relationship list. Only considers "duplicates"!
        Needed before dropping duplicates in some way or another -> preserve MD!

        :param list_rel: Relationship list
        """

        if list_rel is None: return

        for relation in list_rel:
            equivalent_relationships = [rel for rel in list_rel if rel == relation]

            # refresh all "update candidates" ...
            for relation_to_update in equivalent_relationships:
                self.align_resource_metadata(relation_to_update, relation)

        # assumes that once freshest MD is processes, it has been propagated to all other MD!

    def propagate_metadata_in_entities(self, list_ent: list[Entity]):
        """
        Propagates entity metadata in an entity list. Only considers "duplicates"!
        Needed before dropping duplicates in some way or another -> preserve MD!

        :param list_ent: Entity list
        """

        if list_ent is None: return

        for entity in list_ent:
            equivalent_entities = [ent for ent in list_ent if ent == entity]

            # refresh all "update candidates" ...
            for entity_to_update in equivalent_entities:
                self.align_resource_metadata(entity_to_update, entity)

        # assumes that once freshest MD is processes, it has been propagated to all other MD!

    def propagate_metadata_in_entity_lists(self, list_a: list[Entity], list_b: list[Entity]):
        """
        Propagates entity metadata in two entity lists. Only considers "duplicates"!
        Needed, as matching does not handle MD of duplicates in either list (to be matched as well as replacements!)

        :param list_a: First entity list
        :param list_b: Second entity list
        """

        if list_a is None or list_b is None: return

        for entity in list_a:
            equivalent_entities = [ent for ent in list_a if ent == entity]
            equivalent_entities.extend([ent for ent in list_b if ent == entity])

            # refresh all "update candidates" ...
            for entity_to_update in equivalent_entities:
                self.align_resource_metadata(entity_to_update, entity)

        for entity in list_b:
            equivalent_entities = [ent for ent in list_b if ent == entity]
            equivalent_entities.extend([ent for ent in list_a if ent == entity])

            # refresh all "update candidates" ...
            for entity_to_update in equivalent_entities:
                self.align_resource_metadata(entity_to_update, entity)

        # assumes that once freshest MD is processes, it has been propagated to all other MD!

    def process_lists(self,
                      list1: list[Union[Entity, Relationship]],
                      list2: list[Union[Entity, Relationship]],
                      threshold: float = 0.8,
                      protected: list[str] = None,
                      propagate_md = False
                      ) -> Tuple[list[Union[Entity, Relationship]], list[Union[Entity, Relationship]]]:
        """
        Processes two lists to generate new lists based on specified conditions.

        :param list1: First list to process (local items)
        :param list2: Second list to be compared against (global items)
        :param threshold: Cosine similarity threshold (defaults to 0.8)
        :param protected: Entity types and relationships to spare from being replaced or
            being the replacement; especially useful for special objects like experts, topics etc. (defaults to None)
        :param propagate_md: Propagate entity (!) metadata after processing (defaults to False)
        :returns: (matched_local_items, new_global_items)
        """

        list3 = [self.find_match(obj1, list2, threshold=threshold, protected=protected) for obj1 in list1]  # matched_local_items
        if propagate_md: self.propagate_metadata_in_entity_lists(list2, list3)
        list4 = self.create_union_list(list3, list2)  # new_global_items
        return list3, list(set(list4))

    @staticmethod
    def create_union_list(list1: list[Union[Entity, Relationship]], list2: list[Union[Entity, Relationship]]) -> list[
        Union[Entity, Relationship]]:
        """
        Creates a union of two lists (Entity or Relationship objects), avoiding duplicates.
        If it's a relationship, matching will be based on the relationship's name.
        If it's a Entity, matching will be based on both the Entity's name and label.

        :param list1: First list of Entity or Relationship objects
        :param list2: Second list of Entity or Relationship objects
        :returns: Unified list of Entity or Relationship objects
        """

        union_list = list1.copy()

        # Store existing names and labels in the union list
        existing_entity_key = {(obj.name, obj.label) for obj in union_list if isinstance(obj, Entity)}
        existing_relation_names = {obj.name for obj in union_list if isinstance(obj, Relationship)}

        for obj2 in list2:
            if isinstance(obj2, Entity):
                # For Entities, check both name and label to avoid duplicates
                if (obj2.name, obj2.label) not in existing_entity_key:
                    union_list.append(obj2)
                    existing_entity_key.add((obj2.name, obj2.label))

            elif isinstance(obj2, Relationship):
                # For relationships, check based on the name only
                if obj2.name not in existing_relation_names:
                    union_list.append(obj2)
                    existing_relation_names.add(obj2.name)

        return union_list

    def match_entities_and_update_relationships(
            self,
            entities1: list[Entity],
            entities2: list[Entity],
            relationships1: list[Relationship],
            relationships2: list[Relationship],
            rel_threshold: float = 0.8,
            ent_threshold: float = 0.8,
            protected_entities_and_relations: list[str] = None
    ) -> Tuple[list[Entity], list[Relationship]]:
        """
        Matches two lists of entities (Entities) and update the relationships list accordingly.

        :param entities1: First list of entities to match
        :param entities2: Second list of entities to match against
        :param relationships1: First list of relationships to update
        :param relationships2: Second list of relationships to compare
        :param rel_threshold: Cosine similarity threshold for relationships (defaults to 0.8)
        :param ent_threshold: Cosine similarity threshold for entities (defaults to 0.8)
        :param protected_entities_and_relations: Entity types and relationships to spare from being replaced or
            being the replacement; especially useful for special objects like experts, topics etc. (defaults to None)
        :returns: Updated entities list and relationships list
        """

        # Step 1: Match the entities and relations from both lists
        matched_entities1, global_entities = self.process_lists(entities1, entities2, ent_threshold, protected_entities_and_relations, True)
        self.propagate_metadata_from_entities_to_relationships(global_entities, relationships2)
        self.propagate_metadata_from_entities_to_relationships(matched_entities1, relationships1)
        matched_relations, _ = self.process_lists(relationships1, relationships2, rel_threshold, protected_entities_and_relations)

        # Step 2: Create a mapping from old entity names to matched entity names
        entity_name_mapping = {
            entity: matched_entity
            for entity, matched_entity in zip(entities1, matched_entities1)
            if entity != matched_entity
        }

        # Step 3: Update relationships based on matched entities
        def update_relationships(relationships: list[Relationship]) -> list[Relationship]:
            updated_relationships = []
            for rel in relationships:
                updated_rel = rel.model_copy()  # Create a copy to modify
                # Update the 'startEntity' and 'endEntity' names with matched entity names
                if rel.startEntity in entity_name_mapping:
                    updated_rel.startEntity = entity_name_mapping[rel.startEntity]
                if rel.endEntity in entity_name_mapping:
                    updated_rel.endEntity = entity_name_mapping[rel.endEntity]
                updated_relationships.append(updated_rel)
            return updated_relationships

        # Step 4: Extend relationships2 with updated relationships
        global_relationships = [relationship for relationship in relationships2]
        global_relationships.extend(update_relationships(matched_relations))
        # relationships2.extend(update_relationships(matched_relations)) # falsely extends object of existing of kg ...

        return global_entities, global_relationships
