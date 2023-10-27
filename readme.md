# Knowledge Structure Python 

## How to run
0. `pip install -r requirement.txt`
1. prepare text file in .txt format
2. setup parameters in `visualization.py`
3. run `python3 visualization.py`
4. check graph on `http://127.0.0.1:8050/`

## load_data.py
- `read_txt(name)`:
  - **Description**: This function reads a txt file. If there's a file named `abc.txt` in the `data/txt` folder, you input `abc` to `name` to read it.
  - **Details**: The `f.readlines` method reads the txt document line by line, with each line being an entry in the `paragraphs` list (e.g., `['First_line', 'Second_line', ...]`).

## preprocess.py
- **Functions**:
  - `stopwords`: Add desired stopwords for removal. Used in `clean_stopwords`.
  - `mecab_tokenize(sent)`: Uses the `mecab` tokenizer for Korean.
  - `clean_txt(str)`: Removes special characters from a string.
  - `clean_stopword(tokens)`: Removes stopwords from tokenized words.
  - `preprocessing(paragraph)`: Comprehensive preprocessing function.
  - `preprocess_node(paragraphs)`: Preprocesses paragraphs.
  - `preprocess_edge(paragraphs)`: Preprocesses down to the sentence level.
  - `make_dic_tfidf(list_of_wordlist)`: Generates a TF-IDF dictionary.
  - `make_dic_count(tokens)`: Counts term frequencies.
  - `stcs_dic_count(ListOfSentence)`: Calculates term frequencies at the sentence level.

## graph.py
- **KnowledgeGraph class and related functions**:
  - `__init__`: Initializes graph parameters.
  - `get_nodes`: Returns a dictionary of nodes sorted by TF-IDF/TextRank score.
  - `get_adj`: Returns an adjacency matrix.
  - `make_graph`: Creates a graph using the NetworkX library.
  - `cutting_edge(graph, adj, r, option)`: Prunes unnecessary edges from the graph.

## visualization.py
- **Description**: Visualizes the graph using Python Dash.
- **Usage**: Adjust the title in the `visualization.py` file. For further customization, refer to the Dash documentation ([Dash Docs](https://dash.plotly.com/) and [Dash Cytoscape](https://dash.plotly.com/cytoscape)).

## util.py
- **Functions**:
  - `correlation(adj)`: Measures the correlation between two nodes in a graph.
  - `user_define_dict(dict)`: Allows users to define their own nodes...

(Note: Some descriptions were truncated for brevity.)
