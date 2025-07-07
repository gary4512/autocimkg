from ..utils import LLMIntegrator


class DocumentsDistiller:
    """
    Designed to distill essential information from multiple documents into a combined
    structure, using natural language processing tools to extract and consolidate information.
    """

    def __init__(self, llm_model):
        """
        Initializes the DocumentsDistiller with a specified language model.
        
        :param llm_model: Language model instance to be used for generating semantic blocks
        """

        self.llm_integrator = LLMIntegrator(llm_model=llm_model, embeddings_model=None)

    @staticmethod
    def __combine_dicts(dict_list: list[dict]) -> dict:
        """
        Combines a list of dictionaries into a single dictionary, merging values based on their types.
        
        :param dict_list: List of dictionaries to combine
        :returns: Combined dictionary with merged values
        """

        combined_dict = {}

        for d in dict_list:
            for key, value in d.items():
                if key in combined_dict:
                    if isinstance(value, list) and isinstance(combined_dict[key], list):
                        combined_dict[key].extend(value)
                    elif isinstance(value, str) and isinstance(combined_dict[key], str):
                        if value and combined_dict[key]:
                            combined_dict[key] += f' {value}'
                        elif value:
                            combined_dict[key] = value
                    elif isinstance(value, dict) and isinstance(combined_dict[key], dict):
                        combined_dict[key].update(value)
                    else:
                        combined_dict[key] = value
                else:
                    combined_dict[key] = value

        return combined_dict

    def distill(self, documents: list[str], document_type: str, output_data_structure) -> dict:
        """
        Distills information from multiple documents based on a specific information extraction query.
        
        :param documents: List of documents from which to extract information
        :param document_type: Type of document to be distilled (e.g. 'scientific article')
        :param output_data_structure: Data structure definition for formatting the output JSON
        :returns: Dictionary representing distilled information from all documents
        """

        instructions = "# DIRECTIVES\n" \
                     + "- Act like an experienced information extractor.\n" \
                    + f"- You have a chunk of a {document_type}.\n" \
                     + "- If you do not find the right information, keep its place empty.\n" \
                     + "- Translate all parts of the result to English, if they are in another language."

        output_jsons = list(
            map(
                lambda context: self.llm_integrator.extract_information(
                    instructions=instructions,
                    context= f"# CONTEXT\n{context}",
                    output_data_structure=output_data_structure
                ),
                documents))

        return DocumentsDistiller.__combine_dicts(output_jsons)
