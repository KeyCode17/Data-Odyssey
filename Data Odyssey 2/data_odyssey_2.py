# Streamlit Core
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Libraries
import os
import sys

# Import DO2
import re  
# Standard library imports
import os
import re  
import pickle
import warnings

# Data manipulation and numerical computations
import numpy as np  
import pandas as pd  

# Data visualization
import plotly.express as px  
import plotly.graph_objs as go  
from plotly.subplots import make_subplots  

# Machine Learning and Text Processing
from sklearn.manifold import TSNE  
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups  
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Natural Language Processing - NLTK
import nltk
from nltk.corpus import stopwords
from nltk.chunk import tree2conlltags 
from nltk.stem import WordNetLemmatizer 
from nltk.chunk.named_entity import Maxent_NE_Chunker
from nltk.tokenize import word_tokenize, sent_tokenize

# Topic Modeling and Visualization
import pyLDAvis
import pyLDAvis.lda_model

# Additional NLP Tools
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

# Deep Learning - Transformers
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cpu')
dirloc = os.path.dirname(os.path.abspath(__file__))

def data_odyssey_2():
    # Set the working directory
    data_loc = os.path.join(dirloc, 'assets')
    pyLDAvis_loc = os.path.join(dirloc, 'assets', 'pyLDAvis.html')
    wordcloud_loc = os.path.join(dirloc, 'assets', 'wordcloud.html')
    analysis_loc = os.path.join(dirloc, 'assets', 'analysis_results.pkl')
    model_loc = os.path.join(dirloc, 'model_artifacts', 'fine_tuned_model')
    topic_landscape_loc = os.path.join(dirloc, 'assets', 'topic_landscape.html')

    # Load the cleaned texts
    cleaned_texts = pd.read_csv(os.path.join(data_loc, 'cleaned_texts.csv'))
    
    # Load the analysis results
    with open(analysis_loc, 'rb') as file:
        analysis_results = pickle.load(file)

    # Load the tokenizer  
    tokenizer = GPT2Tokenizer.from_pretrained(model_loc)  

    # Set the pad token if not already set  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  

    # Set into CPU
    device = torch.device('cpu')

    # Load the model  
    model = GPT2LMHeadModel.from_pretrained(model_loc).to(device) 

    st.title("Data Odyssey 2")

    option = st.selectbox(
        "Choose Section",
        ("Project Domain", "Business Understanding", "Data Understanding", "Exploratory Data Analysis", "AI Chat Assistant", "Conclusion"),
    )

    st.query_params["Section"]=option

    if st.query_params["Section"] == 'Project Domain':
        st.header("Project Domain", divider="gray")
        st.markdown("This project explores text analysis and generation using the 20 Newsgroups dataset through advanced natural language processing (NLP) techniques. The dataset comprises a diverse collection of news articles spanning multiple topics, providing a rich resource for computational text analysis and generative modeling.")

        st.header("Why This Problem Matters")
        st.markdown('''
            Text analysis and text generation have many applications in various fields, including social media analysis,
            information retrieval, and chatbot development. Solving this problem is important because:
            1. Better Understanding of Context: Helps in understanding the topic contained in the text.
            2. Sentiment and Opinion: Provides insight into the sentiment contained in text for applications such as marketing or social media analysis.
            3. Text Generation: Generates relevant text that can be used for various automated applications.
        ''')

        st.header("References")
        st.markdown('''
        1. Wang, Y. (2024). Research on the TF–IDF algorithm combined with semantics for automatic extraction of keywords from network news texts. Journal of Intelligent Systems. https://doi.org/10.1515/jisys-2023-0300.
        ''')

    if st.query_params["Section"] == 'Business Understanding':
        st.header("Business Understanding", divider="gray")
        st.header("Problem Statements")
        st.markdown('''
            1. How can we effectively categorize and understand diverse textual content across multiple news group domains using advanced natural language processing techniques?
            2. What are the underlying semantic patterns and topic structures that emerge from a multi-category text corpus spanning technology, sports, religion, and other discussion groups?
            3. How can machine learning models, specifically topic modeling and text generation approaches, extract meaningful insights and generate contextually relevant text from complex, unstructured newsgroup discussions?
        ''')

        st.header("Goals")
        st.markdown('''
            1. Comprehensive Text Analysis Framework
            2. Advanced Topic Modeling and Semantic Exploration
            3. Intelligent Text Understanding and Generation
        ''')

        st.header("Solution Statement")
        st.subheader("Text Cleaning and Normalization")
        st.markdown('''
            - Implement comprehensive text preprocessing techniques:
                - Lowercase conversion
                - URL and email address removal
                - Special character elimination
                - Advanced tokenization
                - Lemmatization
                - Stopword removal
        ''')

        st.subheader("Feature Extraction Strategies")
        st.markdown('''
            - Develop sophisticated feature extraction methods:
                - TF-IDF vectorization (max_features: 10,000)
                - N-gram feature engineering
                - Dimensionality reduction techniques
        ''')

        st.header("Semantic Pattern Discovery")
        st.subheader("Topic Modeling Techniques")
        st.markdown('''
            - Implement Latent Dirichlet Allocation (LDA) to:
                - Extract 6 distinct topic structures
                - Generate probabilistic topic distributions
                - Achieve high topic coherence (> 0.5)
                - Discover unique linguistic patterns
        ''')

        st.subheader("Visualization and Interpretation")
        st.markdown('''
            - Create advanced visualization methods:
                - t-SNE dimensional reduction
                - Interactive topic landscape exploration
                - Semantic relationship mapping
                - Topic similarity heatmaps
        ''')

        st.header("Machine Learning and NLP Techniques")
        st.subheader("Advanced Analysis Capabilities")
        st.markdown('''
            - Implement multi-modal NLP techniques:
                - Named Entity Recognition (NLTK and SpaCy)
                - Sentiment analysis (TextBlob)
                - Contextual text generation
                - Fine-tuned language models
        ''')

        st.subheader("Generative and Predictive Modeling")
        st.markdown('''
            - Develop models with capabilities to:
                - Generate contextually relevant text
                - Predict semantic structures
                - Provide deep textual insights
        ''')


    if st.query_params["Section"] == 'Data Understanding':
        st.header("Data Understanding", divider="gray")
        st.header("Dataset Overview")
        st.markdown('''
            - Total Documents: 11314
            - Number of Categories: 20
        ''')
        
        st.header("Categories Data")
        st.markdown('''
            0. `alt.atheism`                : Discussions about atheism and religious skepticism
            1. `comp.graphics`              : Computer graphics, rendering, and visualization
            2. `comp.os.ms-windows.misc`    : Microsoft Windows operating system discussions
            3. `comp.sys.ibm.pc.hardware`   : IBM PC compatible hardware discussions
            4. `comp.sys.mac.hardware`      : Apple Macintosh hardware discussions
            5. `comp.windows.x`             : X-Window System discussions
            6. `misc.forsale`               : Items for sale and marketplace discussions
            7. `rec.autos`                  : Automotive discussions and topics
            8. `rec.motorcycles`            : Motorcycle discussions and topics
            9. `rec.sport.baseball`        : Baseball sports discussions
            10. `rec.sport.hockey`          : Hockey sports discussions
            11. `sci.crypt`                 : Cryptography and encryption discussions
            12. `sci.electronics`           : Electronics and electrical engineering topics
            13. `sci.med`                   : Medical and health-related discussions
            14. `sci.space`                 : Space, astronomy, and space exploration
            15. `soc.religion.christian`    : Christian faith and religious discussions
            16. `talk.politics.guns`        : Firearms and gun policy discussions
            17. `talk.politics.mideast`     : Middle Eastern politics and current events
            18. `talk.politics.misc`        : General political discussions
            19. `talk.religion.misc`        : Various religious discussions and debates
        ''')


    if st.query_params["Section"] == 'Exploratory Data Analysis':
        # 1. Calculate Perplexity Score
        perplexity = analysis_results['lda_model'].perplexity(analysis_results['tfidf_matrix'])

        # 2. Calculate Topic Coherence
        def calculate_topic_coherence(model, feature_names, n_top_words=10):
            """
            Calculates the coherence scores for each topic in the LDA model.

            Parameters:
                model (LatentDirichletAllocation): The trained LDA model.
                feature_names (list of str): The list of feature names.
                n_top_words (int): The number of top words to consider for coherence calculation.

            Returns:
                list of float: Coherence scores for each topic.
            """
            coherence_scores = []
            for topic_idx, topic in enumerate(model.components_):
                top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_indices]

                word_coherence = []
                for i in range(len(top_words)):
                    for j in range(i + 1, len(top_words)):
                        # Calculate coherence based on shared characters (simple example)
                        word_coherence.append(1 if len(set(top_words[i]) & set(top_words[j])) > 0 else 0)

                coherence_scores.append(np.mean(word_coherence) if word_coherence else 0)

            return coherence_scores

        # 3. Calculate Topic Diversity
        def calculate_topic_diversity(model, feature_names, n_top_words=10):
            """
            Calculates the diversity of topics based on the uniqueness of their top words.

            Parameters:
                model (LatentDirichletAllocation): The trained LDA model.
                feature_names (list of str): The list of feature names.
                n_top_words (int): The number of top words to consider for diversity calculation.

            Returns:
                dict: A dictionary containing the number of unique words, total words, and diversity ratio.
            """
            all_top_words = []
            for topic in model.components_:
                top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_indices]
                all_top_words.extend(top_words)

            unique_words = len(set(all_top_words))
            total_words = len(all_top_words)

            return {
                'unique_words': unique_words,
                'total_words': total_words,
                'diversity_ratio': unique_words / total_words if total_words > 0 else 0
            }

        # 4. Prepare Topic Summaries for Interpretability
        def prepare_topic_summaries(model, feature_names, n_top_words=10):
            """
            Prepares summaries of each topic, including top words and their corresponding weights.

            Parameters:
                model (LatentDirichletAllocation): The trained LDA model.
                feature_names (list of str): The list of feature names.
                n_top_words (int): The number of top words to include in each summary.

            Returns:
                tuple: A tuple containing a list of topic summaries and data for visualization.
            """
            topic_summaries = []
            topic_data_for_visualization = []

            for topic_idx, topic in enumerate(model.components_):
                top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_indices]
                top_weights = topic[top_words_indices]

                # Prepare data for visualization
                topic_data_for_visualization.extend([
                    {
                        'Topic': f'Topic {topic_idx + 1}',
                        'Word': word,
                        'Weight': weight
                    }
                    for word, weight in zip(top_words, top_weights)
                ])

                # Create topic summary
                topic_summaries.append({
                    'topic_number': topic_idx + 1,
                    'top_words': top_words,
                    'top_weights': top_weights.tolist()
                })

            return topic_summaries, topic_data_for_visualization

        # 5. Calculate Topic Similarity
        def calculate_topic_similarity(model):
            """
            Calculates the cosine similarity between topics based on their word distributions.

            Parameters:
                model (LatentDirichletAllocation): The trained LDA model.

            Returns:
                numpy.ndarray: A matrix representing the similarity between each pair of topics.
            """
            topic_similarities = cosine_similarity(model.components_)
            return topic_similarities
        
        st.header("Exploratory Data Analysis", divider="gray")
        st.markdown("##### Top 10 Words from 6 Category")
        for topic in analysis_results['topic_insights']:
            topic_num = topic['theme_number']
            words = ', '.join(topic['top_words'])
            st.write(f"Category {topic_num-1}: {words}")
        
        vis = option_menu(None, ["Distribution", "Topic Exploration", 'Topic Evaluations'], 
            icons=['bi-bar-chart-line-fill',  'bi-map-fill', 'bi-newspaper'], 
            menu_icon="cast", default_index=0, orientation="horizontal",
            styles={"nav-link": {"font-size": "19px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}})
        
        if vis == "Distribution":
            st.plotly_chart(analysis_results['fig_length'])
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Document Length Distribution
                - **Overview**: The first chart shows the distribution of document lengths across the entire dataset.
                - **Key Observations**:
                  - A significant number of documents are very short, with a peak around 0-100 words.
                  - The distribution is heavily right-skewed, indicating that while most documents are short, there are a few long documents (up to 6000 words).
                  - The presence of outliers suggests that some documents are significantly longer than the majority.
                ''')

            st.plotly_chart(analysis_results['fig_length_filtered'])
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Document Length Distribution (0-500 Words)
                - **Overview**: This zoomed-in view focuses on documents with lengths between 0 and 500 words.
                - **Key Observations**:
                  - The distribution remains right-skewed, with a high frequency of documents in the lower word count range.
                  - The box plot indicates that the median document length is around 100-200 words, with a majority of documents falling below 300 words.
                  - There are still some outliers, but they are less pronounced than in the first chart.
                ''')

            st.plotly_chart(analysis_results['fig_category'])
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### News Group Category Distribution
                - **Overview**: This chart displays the distribution of documents across different news group categories.
                - **Key Observations**:
                  - The number of documents is relatively evenly distributed across categories, with no single category dominating.
                  - Each category has a count of around 400-600 documents, suggesting a balanced dataset.
                  - This uniformity may indicate a well-structured dataset, allowing for comprehensive analysis across various topics.
                ''')
        
        if vis == "Topic Exploration":
            st.header("Topic Landscape")
            with open(topic_landscape_loc, 'r') as f:  
                st.components.v1.html(f.read(), height=450, scrolling=False)
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Topic Landscape
                This chart generates a two-dimensional visualization of topics derived from Latent Dirichlet Allocation (LDA) output. It utilizes t-SNE for dimensionality reduction and Plotly for interactive plotting.

                - **Topic Distribution**: 
                    - The scatter plot reveals how topics are distributed in the two-dimensional space. Clusters of points indicate groups of documents that share similar topics.
                - **Topic Overlap**:
                    - Areas where points of different colors are close together may suggest overlapping topics or shared themes among documents.
                - **Outliers**: 
                    - Points that are isolated from clusters may represent documents that are unique or less representative of the main topics.
                ''')

            st.header("Topic Mapping")
            panel = pyLDAvis.lda_model.prepare(analysis_results['lda_model'], analysis_results['tfidf_matrix'], analysis_results['vectorizer'], mds='tsne')
            html_string = pyLDAvis.prepared_data_to_html(panel)
            st.components.v1.html(html_string, width=1500, height=800, scrolling=False)
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Topic Mapping
                The visualization presents an intertopic distance map generated from a Latent Dirichlet Allocation (LDA) model. It helps in understanding the relationships between different topics derived from the text data.
                - **Topic Clusters**: 
                  - Topics 1 and 2 are relatively close, suggesting they may share similar themes or content. In contrast, Topic 5 appears more isolated, indicating it may cover a distinct subject area.

                - **Salient Terms**:
                  - The top terms for the selected topic (e.g., "window," "game," "drive") suggest a focus on technology or gaming, indicating that the documents associated with this topic likely discuss software, gaming, or computer-related issues.

                - **Topic Relevance**:
                  - The ability to adjust the relevance metric allows for a dynamic exploration of the topics, enabling users to see how different terms contribute to the understanding of each topic.
                ''')

        if vis == 'Topic Evaluations':
            coherence_scores = calculate_topic_coherence(analysis_results['lda_model'], analysis_results['feature_names'], 10)
            topic_diversity = calculate_topic_diversity(analysis_results['lda_model'], analysis_results['feature_names'], 10)
            topic_summaries, topic_data_for_visualization = prepare_topic_summaries(analysis_results['lda_model'], analysis_results['feature_names'], 10)
            topic_similarities = calculate_topic_similarity(analysis_results['lda_model'])

            # Print Evaluation Summary
            st.write("\nLDA Model Evaluation Report")
            st.write(f"Perplexity Score: {perplexity:.2f}")
            st.write(f"Average Topic Coherence: {np.mean(coherence_scores):.4f}")
            st.write("Topic Diversity:")
            st.write(f"  - Unique Words: {topic_diversity['unique_words']}")
            st.write(f"  - Total Words: {topic_diversity['total_words']}")
            st.write(f"  - Diversity Ratio: {topic_diversity['diversity_ratio']:.4f}")

            fig_topic_words = px.bar(
                pd.DataFrame(topic_data_for_visualization),
                x='Word',
                y='Weight',
                color='Topic',
                title='Topic Word Importance',
                labels={'Weight': 'Word Weight'},
                height=600,
                width=1200
            )
            fig_topic_words.update_layout(
                xaxis_tickangle=-45,
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig_topic_words)
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Topic Word Importance
                - **Overview**: This bar chart displays the importance of words associated with each topic.
                - **Key Observations**:
                  - Each color represents a different topic, with bars indicating the weight of specific words.
                  - Some topics have distinct keywords that dominate their representation, suggesting clear thematic focuses.
                  - The distribution of word importance shows that certain topics are characterized by a few highly relevant words, while others have a broader range of significant terms.
                  - This visualization aids in identifying which words are most influential in defining each topic.
                ''')

            fig_topic_similarity = px.imshow(
            topic_similarities,
            title='Topic Similarity Heatmap',
            labels=dict(x="Topics", y="Topics", color="Similarity"),
            color_continuous_scale='Viridis',
            height=800,
            width=800
            )
            fig_topic_similarity.update_layout(
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig_topic_similarity)
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Topic Similarity Heatmap
                - **Overview**: The heatmap illustrates the similarity between topics based on their word distributions.
                - **Key Observations**:
                  - The color intensity indicates the degree of similarity, with darker shades representing higher similarity scores.
                  - Topics that are closer together in the heatmap may share common themes or vocabulary, while those further apart are more distinct.
                  - This visualization helps in identifying potential overlaps between topics, which can inform adjustments to the model or further analysis.
                ''')

            fig_coherence = px.bar(
            x=[f'Topic {i+1}' for i in range(len(coherence_scores))],
            y=coherence_scores,
            title='Topic Coherence Scores',
            labels={'x': 'Topics', 'y': 'Coherence Score'},
            height=500,
            width=800
            )
            fig_coherence.update_layout(
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig_coherence)
            with st.expander("Analysis", expanded=False):
                st.markdown('''
                ### Topic Coherence Scores
                - **Overview**: This bar chart presents the coherence scores for each topic.
                - **Key Observations**:
                  - Coherence scores provide a measure of how semantically related the top words in each topic are.
                  - Higher scores indicate better-defined topics, while lower scores suggest that the topic may be less coherent.
                  - The chart shows variability in coherence across topics, which can guide further refinement of the model to improve weaker topics.
                ''')
        
    if st.query_params["Section"] == 'AI Chat Assistant':
        # Load the tokenizer  
        tokenizer = GPT2Tokenizer.from_pretrained(model_loc)  
        
        # Set the pad token if not already set  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
        
        # Set into CPU
        device = torch.device('cpu')
        
        # Load the model  
        model = GPT2LMHeadModel.from_pretrained(model_loc).to(device)  

        def generate_text(prompt="The", max_length=50):  
            # Encode the input prompt  
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  

            # Generate text  
            with torch.no_grad():  
                output = model.generate(  
                    input_ids,   
                    max_length=max_length,   
                    num_return_sequences=1,  
                    no_repeat_ngram_size=2,  
                    do_sample=True,  
                    top_k=50,  
                    top_p=0.95,  
                    pad_token_id=tokenizer.eos_token_id  
                )  

            # Decode and return the generated text  
            return tokenizer.decode(output[0], skip_special_tokens=True) 

        # Initialize session state for messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Custom CSS for chat styling
        st.markdown("""
        <style>
        .user-message {
            background-color: #e6f3ff;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            max-width: 80%;
            align-self: flex-end;
            text-align: right;
            margin-left: auto;
        }
        .bot-message {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            max-width: 80%;
            align-self: flex-start;
            text-align: left;
            margin-right: auto;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
        }   
        </style>
        """, unsafe_allow_html=True)

        with st.expander("TF-IDF Formula", expanded=False):
            st.latex(r'''
            \text{TF-IDF}(t, d, D) = \left( \frac{\text{Count of } t \text{ in } d}{\text{Total terms in } d} \right) \times \log \left( \frac{|D|}{|\{d \in D : t \in d\}|} \right)
            ''')
            st.markdown('''
            Wang, Y. (2024). Research on the TF–IDF algorithm combined with semantics for automatic extraction of keywords from network news texts. Journal of Intelligent Systems. https://doi.org/10.1515/jisys-2023-0300.
            ''')

        # Title and description
        st.title("🤖 AI Chat Assistant")
        st.write("Chat with your fine-tuned AI model!")

        # Create a container string for the chat
        chat_html = '<div class="chat-container" id="chat-container">'

        # Loop through the messages and build the content
        for message in st.session_state.get("messages", []):
            if message["role"] == "user":
                chat_html += f'<div class="user-message">{message["content"]}</div>'
            else:
                chat_html += f'<div class="bot-message">{message["content"]}</div>'

        # Close the container
        chat_html += '</div>'

        # Display the entire chat container with messages
        st.markdown(chat_html, unsafe_allow_html=True)

        # User input
        user_input = st.text_input("Your message:", key="user_input")

        # Send button
        if st.button("Send"):
            if user_input:
                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Generate bot response
                try:
                    bot_response = generate_text(prompt=user_input, max_length=100)

                    # Add bot message to session state
                    st.session_state.messages.append({"role": "bot", "content": bot_response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

                # Rerun to update the chat display
                st.rerun()

        # Optional: Clear chat history
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    if st.query_params["Section"] == 'Conclusion':
        st.header("Conclusion", divider="gray")
        st.write('This project successfully addressed the challenges outlined in the problem statements, demonstrating the capability of advanced natural language processing (NLP) techniques to analyze and generate meaningful insights from text data within the 20 Newsgroups dataset.')
        st.markdown('''
### Addressing Problem 1: Categorizing and Understanding Diverse Textual Content  
By utilizing Latent Dirichlet Allocation (LDA) for topic modeling alongside effective preprocessing techniques such as tokenization, lemmatization, and stopword removal, the project provided a robust framework for categorizing and understanding textual content. The high coherence of identified topics confirmed the ability of these techniques to structure and interpret diverse discussions spanning across 20 well-defined categories, ensuring both accuracy and contextual relevance in the categorization.

### Addressing Problem 2: Semantic Pattern and Topic Structure Identification  
The analysis revealed latent semantic patterns and relationships in the data set. The piece of work showcased 6 broad/linked topic structures to create meaningful insights on the common subjects in multi domain discussions technology, sports, religion, politics, etc. Strategies such as using topic similarity heatmaps and semantic relationship mapping helped visualize these patterns and highlight overlaps between related topics. For example, debates in fields such as sports and recreation highlighted subtle intersections that closely matched the interests of real-world users.

### Addressing Problem 3: Insights Extraction and Contextual Text Generation  
Recent work in machine learning, such as fine-tuned generative models like GPT-2, showed the ability to generate human-like, context-appropriate text. Not only did this verify that the generative model was doing its job but it offered tools in practice to the steps like automatic content generation and conversational artificial intelligence. Sentiment analysis also provided an emotional lens to this study, allowing deeper insights to be derived.

---

### Key Findings:
1. Topic Modeling Effectiveness:
The topic structures identified by LDA were interpretable and reflected specific linguistic patterns (with coherence scores>0.5), and enabled the successful extraction of six distinct topic structures. This demonstrated the model's ability to extract meaningful and semantically coherent topics in a variety of domains.

2. Dataset Insights:
The dataset included 11,314 documents across 20 categories with observed differences in document length and content distribution (see Table 1). Categories like rec. sport. baseball and talk. politics. misc topics showed striking overlap, suggesting subtlety of discussion within these areas.

3. Generative Modeling:
When you train GPT-2 models with a particular dataset, the contextually relevant and human-like text was achieved. This showed how advanced language models could generate coherent responses for specific prompts.

4. For Interpretation and Visualization:
The availability of such tools as interactive topic maps and semantic similarity visualizations helped interpret the results, allowing assessment of topic relationships, overlaps, and distinctiveness. These tools highlighted the importance of presenting findings in a clear line of message for both technical and non-technical stakeholders.

5. Sentiment and Semantic Patterns:
With sentiment analysis, we explored the tone and emotional characteristics of the dataset showing user opinions and text implications. In domain-oriented discussions, patterns of sentiment polarity offered further levels of insight.

---

### Overall
The project demonstrated how NLP and machine learning models effectively address multifaceted challenges in text analysis. By categorizing diverse content, revealing semantic patterns, and creating contextually relevant outputs, this approach not only answered the defined problem statements but also laid a foundation for innovative applications in fields like AI chat assistants. With these advancements, text analysis has proven to be a powerful tool for extracting value from unstructured data.
''')