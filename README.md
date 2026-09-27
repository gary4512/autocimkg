# AutoCimKG: Automatic Construction and Incremental Maintenance of Knowledge Graphs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15834992.svg)](https://doi.org/10.5281/zenodo.15834992)

AutoCimKG is a reusable, prototypical Python module that automatically builds and incrementally updates knowledge graphs (KGs) about experts and competencies. 
It applies unified, semantic text processing by prompting off-the-shelf large language models (LLMs), such as OpenAI's [GPT-4o](https://openai.com/de-DE/index/hello-gpt-4o/).
For this purpose, the approach utilises the LLM abstraction framework [LangChain](https://www.langchain.com/).
The training-free AutoCimKG consumes unstructured texts and tries to elicit knowledge and skills (as well as their relations) applied in the processed documents and attributable to their authors.
Additionally, it processes relational master data to resolve valid experts and determine associated facts like department affiliation or employment status. 
Consequently, the extensible prototype yields an overall KG that encodes distinct experts, competencies, documents and organisational units across inputs.
This is supported by ongoing entity and relationship resolution (building on text embeddings calculated by LLMs, such as OpenAI's [text-embedding-3-large](https://openai.com/index/new-embedding-models-and-api-updates/)). 
Moreover, an existing KG can be extended incrementally with new documents, if available. No batch-like rebuild is needed for this purpose.
<br/>
<br/>
The developed software system offers even more functionality. AutoCimKG also allows to store and retrieve snapshots of KGs to and from a connected [PostgreSQL/Apache AGE database](https://age.apache.org/), which enables query-based utilisation and makes the KG available for downstream tasks.
Moreover, the system also offers to process a user-defined lightweight ontology (respectively semantic schema). 
Here, specialist subject areas and specific relationship types are definable that align the LLM-based extraction of competencies and their relations from text. 
More specifically, subject areas are semantically linked to subordinate competencies and integrated into the KG (adding a more coarse-grained level for utilisation).
Moreover, the ontology lets the user define a strictness level, which indicates to either filter out incompatible competencies or suggest new subject areas for ontology evolution.
Contrarily, relationship types are used to filter and standardise extracted competency relationships.
Last but not least, AutoCimKG manages comprehensive metadata and realises a relational metadata repository in the connected PostgreSQL/Apache AGE database. 
Thus, it allows to store and retrieve data sources, KG version information, system logs, LLM as well as system configurations and applied ontologies. 
In addition, the assembled property graph encodes embedded (i.e. fact-level) provenance about editors, sources and generation as well as invalidation moments.
<br/>
<br/>
The artefact AutoCimKG was developed in the course of a master's thesis. Further information about the project is provided there.
- **Title**: 'Automatic Construction and Incremental Maintenance of Knowledge Graphs: Encoding Employee Competencies in the Case of the Austrian Financial Market Authority'
- **Author:** Gerhard Lerch
- **Year:** 2025
- **University:** Johannes Kepler Universit&auml;t Linz
## License Notice
This software is based on and includes modified components of the [iText2KG library (v0.0.7)](https://github.com/AuvaLab/itext2kg), which is licensed under the GNU Lesser General Public Library (v2.1).
The extensive changes and enhancements made are reflected in all files of the reused codebase and correspond to the overview given above. 
This software is therefore also licensed under LGPL-2.1 and a copy of the license is provided in the 'LICENSE.txt' file.
<br/>
<br/>
Y. Lairgi, L. Moncla, R. Cazabet, K. Benabdeslem, and P. Cléau, ‘iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models’, in Web Information Systems Engineering – WISE 2024, vol. 15439, M. Barhamgi, H. Wang, and X. Wang, Eds., in Lecture Notes in Computer Science, vol. 15439. , Singapore: Springer, 2025, pp. 214–229. doi: 10.1007/978-981-96-0573-6_16.
## Installation
The recommended way to use AutoCimKG is to download the library from GitHub and make it available in a desired Python project.
This can take the form of a [PyCharm](https://www.jetbrains.com/pycharm/) Python project centered around a [Jupyter Notebook](https://jupyter.org/).
The library needs a chat as well as an embedding model, either hosted (e.g. via OpenAI's [developer platform](https://platform.openai.com/)) or running locally (e.g. via [Ollama](https://ollama.com/)).
Moreover, AutoCimKG connects to a [PostgreSQL/Apache AGE database](https://age.apache.org/age-manual/master/intro/setup.html), if desired. 
Another recommendation is to set up the terminal-based [psql](https://www.postgresql.org/docs/current/app-psql.html) 
and [pgAdmin](https://www.pgadmin.org/) to inspect assembled property graphs as well as associated metadata and to query the competency KG (using SQL and Cypher).
<br/>
<br/>
In general, AutoCimKG was developed and is tested with Python v3.9 and lists all required packages in the 'requirements.txt' file.
## LLM Providers
AutoCimKG accepts any [LangChain](https://www.langchain.com/) chat and embeddings model. 
For common providers, ```create_chat_model()``` and ```create_embeddings_model()``` construct suitable models (incl. JSON output mode where supported):
```python
from autocimkg import create_chat_model, create_embeddings_model

# hosted
llm_model = create_chat_model("openai", "gpt-4o")                       # OPENAI_API_KEY
embeddings_model = create_embeddings_model("openai", "text-embedding-3-large")

# local via Ollama
llm_model = create_chat_model("ollama", "qwen3.5:9b", num_ctx=16384)
embeddings_model = create_embeddings_model("ollama", "nomic-embed-text")

# local via LM Studio, vLLM, llama.cpp, LocalAI, ... (OpenAI-compatible API)
llm_model = create_chat_model("openai_compatible", "<model>", base_url="http://localhost:1234/v1")
```
| Provider            | Chat | Embeddings | Package                  |
|---------------------|------|------------|--------------------------|
| `openai`            | yes  | yes        | `langchain-openai`       |
| `azure_openai`      | yes  | yes        | `langchain-openai`       |
| `openai_compatible` | yes  | yes        | `langchain-openai`       |
| `ollama`            | yes  | yes        | `langchain-ollama`       |
| `anthropic`         | yes  | -          | `langchain-anthropic`    |
| `google_genai`      | yes  | yes        | `langchain-google-genai` |
| `mistralai`         | yes  | yes        | `langchain-mistralai`    |
| `huggingface`       | -    | yes        | `langchain-huggingface`  |

Packages beyond ```langchain-openai``` are listed in 'requirements-optional.txt'. Further arguments are handed to the respective LangChain class. 
```get_model_config()``` describes a model without API keys, e.g. for storing it via ```MetadataIntegrator.create_llm_config()```.
Note that an existing KG can only be maintained with the embeddings model it was built with, as entity and relationship resolution compares embeddings.
Moreover, smaller local models tend to produce invalid JSON more often, which AutoCimKG answers with retries (see ```max_tries``` parameters).
## Usage
An exemplary utilisation of AutoCimKG is provided in the ```tutorial```.
