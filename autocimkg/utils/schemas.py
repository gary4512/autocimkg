from typing import Optional
from pydantic import BaseModel, Field

# ENTITIES & RELATIONSHIPS
class Entity(BaseModel):
    label: str = Field("The type or category of the entity, such as 'Process', 'Technique', 'Data Structure',"
                       " 'Methodology', 'Person', 'Company', 'Law', 'Regulation', etc. Do not limit or focus the"
                       " labeling to the mentioned options, these are just examples. This field helps in classifying"
                       " and organizing entities within the knowledge graph.")
    name: str = Field("The specific name of the entity. It should represent a single, distinct concept and must not be"
                      " an empty string. For example, if the entity is a 'Technique', the name could be 'Neural Networks'.")

class EntitiesExtractor(BaseModel):
    entities : list[Entity] = Field("The relevant entities presented in the context. The entities should encode"
                                             " ONE concept.")

class Relationship(BaseModel):
    startNode: Entity = Field("The starting entity, which is present in the entities list.")
    endNode: Entity = Field("The ending entity, which is present in the entities list.")
    name: str = Field("The predicate that defines the relationship between the two entities. This predicate should"
                      " represent a single, semantically distinct relation.")

class RelationshipsExtractor(BaseModel):
    relationships: list[Relationship] = Field("Based on the provided entities and context, identify the predicates that"
                                              " define relationships between these entities. The predicates should be"
                                              " chosen with precision to accurately reflect the expressed relationships.")

class AlignedEntity(BaseModel):
    name: str = Field("The name of the processed entity.")
    label: str = Field("The label of the processed entity.")
    # ... BOTH INCLUDED: otherwise, LLM sometimes confuses provided label of context and assigned topic!
    topic: str = Field("The assigned topic or 'None'.")

class EntityAlignmentExtractor(BaseModel):
    entities: list[AlignedEntity] = Field(
        "The entities of the provided context, matched with the existing topics or 'None'.")

class NovelAlignedEntity(BaseModel):
    name: str = Field("The name of the processed entity.")
    label: str = Field("The label of the processed entity.")
    # ... BOTH INCLUDED: otherwise, LLM sometimes confuses provided label of context and assigned topic!
    topic: str = Field("The newly created topic.")

class NovelEntityAlignmentExtractor(BaseModel):
    entities: list[AlignedEntity] = Field(
        "The entities of the provided context, matched with the newly created topics.")

class NovelTopic(BaseModel):
    topic: str = Field("The processed topic.")
    description: str = Field("The concise description of the topic.")
    examples: list[str] = Field("The examples for the topic.")

class NovelTopicExtractor(BaseModel):
    topics: list[NovelTopic] = Field("The topics of the provided context, enriched with descriptions and examples.")

class AlignedRelationship(BaseModel):
    list_number: int = Field("The number of the processed triple in the provided list.")
    start_entity: str = Field("The start entity, which is present in the processed triple.")
    relationship_type: str = Field("The assigned relational type from the ontology or 'None'.")
    end_entity: str = Field("The end entity, which is present in the processed triple.")

class RelationshipAlignmentExtractor(BaseModel):
    relationships: list[AlignedRelationship] = Field(
        "The relational triples of the provided context, matched with existing relationship types or 'None'.")

# DOMAIN-SPECIFIC SCHEMAS
class AuthorsOnly(BaseModel):
    authors: list[str] = Field(description="The names of the authors of the text (i.e. Persons)")

class ScientificArticle(BaseModel):
    title: str = Field(description="The title of the scientific article")
    abstract: str = Field(description="The scientific article's full abstract")
    keywords: list[str] = Field(description="The listed keywords of the scientific article")
    authors: list[str] = Field(description="The names of the authors of the scientific article (i.e. Persons)")