# HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in

Varun Gumma♠ Anandhita Raghunath\*♢ Mohit Jain†♠ Sunayana Sitaram†♠

♠Microsoft Corporation ♢University of Washington

varun230999@gmail.com, sunayana.sitaram@microsoft.com

# Abstract

Assessing the capabilities and limitations of large language models (LLMs) has garnered significant interest, yet the evaluation of multiple models in real-world scenarios remains rare. Multilingual evaluation often relies on translated benchmarks, which typically do not capture linguistic and cultural nuances present in the source language. This study provides an extensive assessment of 24 LLMs on real world data collected from Indian patients interacting with a medical chatbot in Indian English and 4 other Indic languages. We employ a uniform Retrieval Augmented Generation framework to generate responses, which are evaluated using both automated techniques and human evaluators on four specific metrics relevant to our application. We find that models vary significantly in their performance and that instruction tuned Indic models do not always perform well on Indic language queries. Further, we empirically show that factual correctness is generally lower for responses to Indic queries compared to English queries. Finally, our qualitative work shows that code-mixed and culturally relevant queries in our dataset pose challenges to evaluated models.

# 1 Introduction

Large Language Models (LLMs) have demonstrated impressive proficiency across various domains. Nonetheless, their full spectrum of capabilities and limitations remains unclear, resulting in unpredictable performance on certain tasks. Additionally, there is now a wide selection of LLMs available. Therefore, evaluation has become crucial for comprehending the internal mechanisms of LLMs and for comparing them against each other.

Despite the importance of evaluation, significant challenges still persist. Many widely-used benchmarks for assessing LLMs are contaminated (Ahuja et al., 2024; Oren et al., 2024; Xu et al., 2024), meaning that they often appear in LLM training data. Some of these benchmarks were originally created for conventional Natural Language Processing tasks and may not fully represent current practical applications of LLMs (Conneau et al., 2018; Pan et al., 2017). Recently, there has been growing interest in assessing LLMs within multilingual and multicultural contexts (Ahuja et al., 2023, 2024; Faisal et al., 2024; Watts et al., 2024; Chiu et al., 2024). Traditionally, these benchmarks were developed by translating English versions into various languages. However, due to the loss of linguistic and cultural context during translation, new benchmarks specific to different languages and cultures are now being created. However, such benchmarks are few in number, and several of the older ones are contaminated in training data (Ahuja et al., 2024; Oren et al., 2024). Thus, there is a need for new benchmarks that can test the abilities of models in real-world multilingual settings.

LLMs are employed in various fields, including critical areas like healthcare. Jin et al. (2024) translate an English healthcare dataset into Spanish, Chinese, and Hindi, and demonstrate that performance declines in these languages compared to English. This highlights the necessity of examining LLMs more thoroughly in multilingual contexts for these important uses.

In this study, we conduct the first comprehensive assessment of multilingual models within a real-world healthcare context. We evaluate responses from 24 multilingual and Indic models using 750 questions posed by users of a health chatbot in five languages (Indian English and four Indic languages). All the models being evaluated function within the same RAG framework, and their outputs are compared to doctor-verified ground truth responses. We evaluate LLM responses on four metrics curated for our application, including factual correctness, semantic similarity, coherence,

- Work done during an internship at Microsoft

† Equal Advising
and conciseness and present leaderboards for each metric, as well as an overall leaderboard. We use human evaluation and automated methods (LLMs-as-a-judge) to compute these metrics by comparing LLM responses with ground-truth reference responses or assessing the responses in a reference-free manner.

Our results suggest that models vary significantly in their performance, with some smaller models outperforming larger ones. Factual Correctness is generally lower for non-English queries compared to English queries. We observe that instruction-tuned Indic models do not always perform well on Indic language queries. Our dataset contains several instances of code-mixed and culturally-relevant queries, which models sometimes struggle to answer. The contributions of our work are as follows:

- We evaluate 24 models (proprietary as well as open weights) in a healthcare setting using queries provided by patients using a medical chatbot. This guarantees that our dataset is not contaminated in the training data of any of the models we evaluate.
- We curate a dataset of queries from multilingual users that spans multiple languages. The queries feature language typical of multilingual communities, such as code-switching, which is rarely found in translated datasets, making ours a more realistic dataset for model evaluation.
- We evaluate several models in an identical RAG setting, making it possible to compare models in a fair manner. The RAG setting is a popular configuration that numerous models are being deployed in for real-world applications.
- We establish relevant metrics for our application and determine an overall combined metric by consulting domain experts - doctors working on the medical chatbot project.
- We perform assessments (with and without ground truth references) using LLM-as-a-judge and conduct human evaluations on a subset of the models and data to confirm the validity of the LLM assessment.

# 2 Related Works

# Healthcare Chatbots in India

Within the Indian context, the literature has documented great diversity in health seeking and health communication behaviors based on gender (Das et al., 2018), varying educational status, poor functional literacy, cultural context (Islary, 2018), stigmas (Wang et al.) etc. This diversity in behavior may translate to people’s use of medical chatbots, which are increasingly reaching hundreds of Indian patients at the margins of the healthcare system (Mishra et al., 2023). These bots solicit personal health information directly from patients in their native Indic languages or in Indic English. For example, (Ramjee et al., 2024) find that their CataractBot deployed in Bangalore, India yields patient questions on topics such as surgery, preoperative preparation, diet, exercise, discharge, medication, pain management, etc. Mishra et al. (2023) find that Indian people share “deeply personal questions and concerns about sexual and reproductive health” with their chatbot SnehAI. Yadav et al. (2019) find that queries to chatbots are “embedded deeply into a communities myths and existing belief systems” while (Xiao et al., 2023) note that patients have difficulties finding health information at an appropriate level for them to comprehend. Therefore, LLMs powering medical chatbots in India and other Low and Middle Income Countries are challenged to respond lucidly to medical questions that are asked in ways that may be hyperlocal to patient context. Few works have documented how LLMs react to this linguistic diversity in the medical domain. Our work begins to bridge this gap.

# Multilingual and RAG evaluation

Several previous studies have conducted in-depth evaluation of Multilingual capabilities of LLMs by evaluating across standard tasks (Srivastava et al., 2022; Liang et al., 2023; Ahuja et al., 2023, 2024; Asia et al., 2024; Lai et al., 2023; Robinson et al., 2023), with a common finding that current LLMs only have a limited multilingual capacity. Other works (Watts et al., 2024; Leong et al., 2023) include evaluating LLMs on creative and generative tasks. Salemi and Zamani (2024) state that evaluating RAG models require a joint evaluating of the retrieval and generated output. Recent works such as Chen et al. (2024); Chirkova et al. (2024) benchmark LLMs as RAG models in bilingual and multilingual setups. Lastly, several tools and benchmarks have also been built for automatic evaluation of RAG.
even in medical domains (Es et al., 2024; Tang and Yang, 2024; Xiong et al., 2024a,b), and we refer the readers to Yu et al. (2024) for such a comprehensive list and survey.

# LLM-based Evaluators

With the advent of large-scale instruction following capabilities in LLMs, automatic evaluations with the help of these models is being preferred (Kim et al., 2024a,b; Liu et al., 2024; Shen et al., 2023; Kocmi and Federmann, 2023). However, it has been shown that it is optimal to assess these evaluations in tandem with human annotations as LLMs can provide inflated scores (Hada et al., 2024b,a; Watts et al., 2024). Other works (Zheng et al., 2023; Watts et al., 2024) have employed GPT-4 alongside human evaluators to leaderboards to assess other LLMs. Ning et al. (2024) proposed an innovative approach using LLMs for peer review, where models evaluate each other’s outputs. However, a recent study by Doddapani et al. (2024) highlighted the limitations of LLM-based evaluators, revealing their inability to reliably detect subtle drops in input quality during evaluations, raising concerns about their precision and dependability for fine-grained assessments. In this work, we use LLM-based evaluators both with and without ground-truth references and also use human evaluation to validate LLM-based evaluation.

# 3 Methodology

In this study, we leveraged a dataset collected from a deployed medical chatbot. Here, we provide an overview of the question dataset, the knowledge base employed for answering those questions, the process for generating responses, and the evaluation framework.

# 3.1 Data

The real-world test data was collected by our collaborators as part of an ongoing research effort that designed and deployed a medical chatbot, hereafter referred to as HEALTHBOT, to patients scheduled for cataract surgery at a large hospital in urban India. An Ethics approval was obtained from our institution prior to conducting this work, and once enrolled in the study and consent was obtained, both the patient and their accompanying family member or attendant were instructed on how to use HEALTHBOT on WhatsApp. Through this instructional phase, they were informed that questions could be asked by voice or by text, in one of 5 languages - English, Hindi, Kannada, Tamil, Telugu. The workflow of chatting with HEALTHBOT was as follows: Patients sent questions through the WhatsApp interface to HEALTHBOT. Their questions were transcribed automatically (using a speech recognition system) and translated (using an off-the-shelf translator) into English if needed, after which GPT-4 was used to produce an initial response by performing RAG on the documents in the knowledge base (KB, see below). This initial response was passed to doctors who reviewed, validated, and if needed, edited the answer. The doctor approved answer is henceforth referred to as the ground truth (GT) response associated with the patient query.

Our evaluation dataset was curated from this data by including all questions sent to HEALTHBOT along with their associated GT response. Exclusion criteria removed exact duplicate questions, those with personally identifying information, and those not relevant to health. Additionally, for this work, we only consider questions to which the GPT-4 answer was directly approved by the expert as the “correct and complete answer" without additional editing on the doctors’ part. The final dataset contained 749 question and GT answer pairs that were sent in to HEALTHBOT between December 2023 to June 2024. In the pool, 666 questions were in English, 19 in Hindi, 27 in Tamil, 14 in Telugu, and 23 in Kannada. Note that, queries written in the script of a specific language were classified as belonging to that language. For code-mixed and Romanized queries, we determined whether they were English or non-English based on the matrix language of the query.

The evaluation dataset consists of queries that (1) have misspelled English words, (2) are code-mixed, (3) represent non-native English, (4) are relevant to the patient’s cultural context and (5) are specific to the patient’s condition. We provide some examples of each of these categories.

Examples of misspelled queries include questions such as “How long should saving not be done after surgery?” where the patient intended to ask about shaving, and “Sarjere is don mam?” which the attendant used to inquire about the patient’s discharge status. Instances of code mixing can be seen in phrases like “Agar operation ke baad pain ho raha hai, to kya karna hai?” meaning “If there is pain after the surgery, what should I do?” in Hindi-English (Hinglish). Other examples include “Can I eat before the kanna operation?” where
“kanna” means eye in Tamil, and “kanna operation” is a well understood, common way of referring to cataract surgery, and “In how many days can a patient take Karwat?” where “Karwat” means turning over in sleep in Hindi.

# 3.3 Models

Indian English was used in a majority of the English queries, making the phrasing of questions different from what they would be with native English speech. Examples are as follows - “Because I have diabetes sugar problem I am worried much”, “Why to eat light meal only? What comes under light meal?” and “Is the patient should be in dark room after surgery?” Taking a shower was commonly referred to as “taking a bath”, and eye glasses were commonly referred to as “goggles”, “spex” or “spectacles”.

Culturally-relevant questions were also many in number, for example questions about specific foods were asked like “Can he take chapati, Puri etc on the day of surgery?” and “Can I eat non veg after surgery?” (“non-veg” is a term used in Indian English to denote eating meat). Questions about yoga were asked, like “How long after the surgery should the Valsalva maneuver be avoided?” and “Are there any specific yoga poses I can do?”. The notion of a patient’s native place or village was brought up in queries such as “If a person gets operated here and then goes to his native place and if some problem occurs what shall he do?” or “Can she travel by car with AC for 100 kms?”.

# 3.4 Response Generation

We chose 24 models including proprietary multilingual models, as well as Open-weights multilingual and Indic language models for our evaluation. A full list of models can be found in Table 1.

We use the standard Retrieval-Augmented-Generation (RAG) strategy to elicit responses from all the models. Each model is asked to respond to the given query by extracting the appropriate pieces of text from the knowledge-base chunks. During prompting, we segregate the chunks into RAWCHUNKS and KBUPDATECHUNKS symbolizing the data from the standard sources, and the KB updates. Then model is explicitly instructed to prioritize the information from the most latest sources, i.e. the KBUPDATECHUNKS (if they are available). The exact prompt used for generation is provided in Appendix X. Note that each model gets the same RAWCHUNKS and KBUPDATECHUNKS, which are also the same that are given to the GPT-4 model in the HEALTHBOT, based on which the GT responses are verified.

# 3.5 Response Evaluation

We used both human and automated evaluation to evaluate the performance of models in the setup described above. GPT-4o3 was employed as an LLM evaluator. We prompted the model separately to judge each metric, as Hada et al. (2024b,a) show that individual calls reduce interaction and influence among and their evaluations.

# 3.5.1 LLM Evaluation

In consultation with domain experts working on the HEALTHBOT, we curated metrics that are relevant for our application. We limit ourselves to 3 classes (Good - 2, Medium - 1, Bad - 0) for each metric, as a larger number of classes could hurt interpretability and lower LLM-evaluator performance. The prompt used for each of our metrics are available in Appendix A.2, and a general overview is provided below.

1 https://www.trychroma.com

2 https://platform.openai.com/docs/guides/embeddings/embedding-models

3 https://openai.com/index/hello-gpt-4o/

# Models

- GPT-4
- GPT-4o
- microsoft/Phi-3.5-MoE-instruct
- CohereForAI/c4ai-command-r-plus-08-2024
- Qwen/Qwen2.5-72B-Instruct
- CohereForAI/aya-23-35B
- mistralai/Mistral-Large-Instruct-2407
- google/gemma-2-27b-it
- meta-llama/Meta-Llama-3.1-70B-Instruct
- GenVRadmin/llama38bGenZ_Vikas-Merged
- GenVRadmin/AryaBhatta-GemmaOrca-Merged
- GenVRadmin/AryaBhatta-GemmaUltra-Merged
- GenVRadmin/AryaBhatta-GemmaGenZ-Vikas-Merged
- Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0
- ai4bharat/Airavata
- Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1
- BhabhaAI/Gajendra-v0.1
- manishiitg/open-aditi-hi-v4
- abhinand/tamil-llama-7b-instruct-v0.2
- abhinand/telugu-llama-7b-instruct-v0.1
- Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Instruct
- Tensoic/Kan-Llama-7B-SFT-v0.5
- Cognitive-Lab/Ambari-7B-Instruct-v0.2
- GenVRadmin/Llamavaad

# Languages Availability

| All    | Proprietary  |
| ------ | ------------ |
| All    | Proprietary  |
| All    | Open-weights |
| All    | Open-weights |
| All    | Open-weights |
| All    | Open-weights |
| All    | Open-weights |
| All    | Open-weights |
| All    | Indic        |
| All    | Indic        |
| All    | Indic        |
| All    | Indic        |
| All    | Indic        |
| En, Hi | Indic        |
| En, Hi | Indic        |
| En, Hi | Indic        |
| En, Hi | Indic        |
| En, Ta | Indic        |
| En, Te | Indic        |
| En, Te | Indic        |
| En, Ka | Indic        |
| En, Ka | Indic        |
| En, Hi | Indic        |

Table 1: List of models tested. “En” for English, “Hi” for Hindi, “Ka” for Kannada, “Ta” for Tamil, “Te” for Telugu, and “All" refers to all the aforementioned languages. All Indic models are open-weights.

# Metrics

- FACTUAL CORRECTNESS (FC): As Doddapa-neni et al. (2024) had shown that LLM-based evaluators fail to identify subtle factual inaccuracies, we curate a separate metric to double-check facts like dates, numbers, procedure and medicine names.
- SEMANTIC SIMILARITY (SS): Similarly, we formulate another metric to specifically analyse if both the prediction and the ground-truth response convey the same information semantically, especially when they are in different languages.
- COHERENCE (COH): This metric evaluates if the model was able to stitch together appropriate pieces of information from the three data chunks provided to yield a coherent response.
- CONCISENESS (CON): Since the knowledge base chunks extracted and provided to the model can be quite large, with important facts embedded at different positions, we build this metric to assess the ability of the model to extract and compress all these bits of information relevant to the query into a crisp response.

# 3.5.2 Human Evaluation

Following previous works (Hada et al., 2024b,a; Watts et al., 2024), we augment the LLM evaluation with human evaluation and draw correlations between the LLM evaluator and human evaluation for a subset of the models (PHI-3.5-MOE-INSTRUCT, MISTRAL-LARGE-INSTRUCT-2407, GPT-4O, META-LLAMA-3.1-70B-INSTRUCT, INDIC-GEMMA-7B-FINETUNED-SFT-NAVARASA-2.0). These models were selected based on results from early automated evaluations, covering a range of scores and representing models of interest.

The human annotators were employed by
KARYA, a data annotation company and were all native speakers of Indian languages that we evaluated. We selected a sample of 100 queries from English, and all the queries from Indic languages for annotation, yielding a total of 183 queries. Each instance was annotated by one annotator for SEMANTIC SIMILARITY between the model’s response and the GT response provided by the doctor. The annotations began with a briefing about the task and each of them was given a sample test task, and were provided some guidance based on their difficulties and mistakes. Finally, the annotators were asked to evaluate the model response based on the metric4, query, and ground-truth response on a scale of 0 to 2, similar to the LLM-evaluator.

# 4 Results

In this section, we present the outcomes of both the LLM and human evaluations. We begin by examining the average scores across all our metrics including the combined metric for English queries, followed by results for queries in other languages. Next, we examine the ranking of models based on the human and LLM-evaluator, details of which can be found in the Appendix A.1 and find it to be consistently higher than 0.7 on average across all languages and models. This shows the reliability of our LLM-based evaluation for SEMANTIC SIMILARITY which uses the GT response as a reference.

# 4.1 LLM evaluator results

We see from Table 2 that for English, the best performing models is the QWEN2.5-72B-INSTRUCT model across all metrics. Note that it is expected that GPT-4 performs well, as the ground truth responses are based on responses generated by GPT-4. The PHI-3.5-MOE-INSTRUCT model also performs well on all metrics, followed by MISTRAL-LARGE-INSTRUCT-2407 and OPEN-ADITI-HI-V4, which is the only Indic model that performs near the top even for English queries. Surprisingly, the META-LLAMA-3.1-70B-INSTRUCT model performs worse than expected on this task, frequently regurgitating the entire prompt that was provided. In general, all models get higher scores on conciseness and many models do well on coherence.

# 4.2 Comparison of human and LLM evaluators

We perform human evaluation on five models on the SEMANTIC SIMILARITY (SS) task and compare human and LLM evaluation by inspecting the ranking of the models in Appendix A.3. We find that for all languages except Telugu, we get identical rankings of all models. Additionally, we also measure the Percentage Agreement (PA) between the human and LLM-evaluator.

| Model                                                                                                                             | English                                                                                                                           | English                                                                                                                           |     |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
|                                                                                                                                   | gpt-4o                                                                                                                            | Indic-gemma-7b-finetuned-sft-Navarasa-2.0                                                                                         |
| Mistral-Large-Instruct-2407                                                                                                       | 0.0                                                                                                                               | 0.1                                                                                                                               |
| Meta-Llama-3.1-70B-Instruct                                                                                                       | 0.2                                                                                                                               | 0.3                                                                                                                               |
| Phi-3.5-MoE-instruct                                                                                                              | 0.4                                                                                                                               | 0.5                                                                                                                               |
| Figure 1: Percentage Agreement between human and LLM-evaluators for English. The red line indicates the average PA across models. | Figure 1: Percentage Agreement between human and LLM-evaluators for English. The red line indicates the average PA across models. | Figure 1: Percentage Agreement between human and LLM-evaluators for English. The red line indicates the average PA across models. |     |     |

# 4.3 Qualitative Analysis

One of the authors of the paper performed a qualitative analysis of responses from the evaluated LLMs on 100 selected patient questions. The questions were chosen to cover a range of medical topics and languages. Thematic analysis involved (1) initial familiarization with the queries and associated LLM responses, (2) theme identification where 5 themes were generated and (3) thematic coding where the generated themes were applied to the 100 question-answer pairs. We briefly summarize these results below.

# Table 2: Metric-wise scores for English. The Proprietary, Open-Weights and Indic models are highlighted appropriately. All Indic models are open-weights.

| Model                                     | AGG  | COH  | CON  | FC   | SS   |
| ----------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| QWEN2.5-72B-INSTRUCT                      | 1.46 | 1.86 | 1.96 | 1.62 | 1.43 |
| GPT-4                                     | 1.40 | 1.71 | 1.95 | 1.56 | 1.36 |
| PHI-3.5-MOE-INSTRUCT                      | 1.29 | 1.65 | 1.93 | 1.43 | 1.22 |
| MISTRAL-LARGE-INSTRUCT-2407               | 1.29 | 1.60 | 1.95 | 1.42 | 1.24 |
| OPEN-ADITI-HI-V4                          | 1.27 | 1.69 | 1.85 | 1.37 | 1.22 |
| LLAMAVAAD                                 | 1.16 | 1.34 | 0.97 | 1.36 | 1.20 |
| ARYABHATTA-GEMMAGENZ-VIKAS-MERGED         | 1.12 | 1.48 | 1.65 | 1.22 | 1.07 |
| KAN-LLAMA-7B-SFT-V0.5                     | 1.01 | 1.39 | 1.64 | 1.07 | 0.97 |
| GEMMA-2-27B-IT                            | 1.00 | 1.28 | 1.88 | 1.07 | 0.91 |
| ARYABHATTA-GEMMAORCA-MERGED               | 0.97 | 1.32 | 1.62 | 1.03 | 0.92 |
| LLAMA3-GAJA-HINDI-8B-V0.1                 | 0.91 | 0.63 | 1.65 | 1.09 | 0.98 |
| GPT-4O                                    | 0.91 | 1.08 | 1.78 | 0.98 | 0.87 |
| AYA-23-35B                                | 0.91 | 1.09 | 1.65 | 1.00 | 0.83 |
| GAJENDRA-V0.1                             | 0.88 | 1.21 | 1.38 | 0.93 | 0.85 |
| C4AI-COMMAND-R-PLUS-08-2024               | 0.82 | 1.15 | 1.48 | 0.85 | 0.74 |
| TAMIL-LLAMA-7BINSTRUCT-V0.2               | -    | 0.81 | 1.13 | 1.50 | 0.83 |
| AIRAVATA                                  | 0.80 | 1.03 | 1.38 | 0.85 | 0.78 |
| AMBARI-7B-INSTRUCTV0.2                    | -    | 0.73 | 0.86 | 1.11 | 0.76 |
| META-LLAMA-3.1-70B-INSTRUCT               | 0.65 | 0.55 | 1.12 | 0.77 | 0.67 |
| TELUGU-LLAMA2-7B-V0-INSTRUCT              | 0.51 | 0.60 | 1.12 | 0.53 | 0.53 |
| LLAMA38BGENZ_VIKAS-MERGED                 | 0.51 | 0.52 | 1.09 | 0.55 | 0.53 |
| INDIC-GEMMA-7B-FINETUNED-SFT-NAVARASA-2.0 | 0.35 | 0.32 | 0.53 | 0.40 | 0.39 |
| ARYABHATTA-GEMMAULTRA-MERGED              | 0.32 | 0.38 | 1.19 | 0.31 | 0.27 |
| TELUGU-LLAMA-7B-INSTRUCTV0.1              | -    | 0.04 | 0.00 | 0.58 | 0.03 |

(1) misspelling of English words, (2) code-mixing, (3) non-native English, (4) relevance to cultural context and (5) specificity to the patient’s condition.

For queries that involve misspellings (such as “saving” and “sarjere” mentioned in Section 3.1), many evaluated LLM were not able to come up with an appropriate response. For the query with the word “saving", responses varied from “The patient should not be saved for more than 15 days after the surgery” to “Saving should not be done after surgery” to “You should not strain to pass motion for 15 days after the surgery. If you are constipated, it is recommended to consult the doctor”. All of these responses deviate from the GPT-4 generated GT, which said “You can have a shave after the cataract surgery. However, you should avoid having a head bath or shampoo for 15 days post-surgery.”

In cases of code mixing and Indian English, LLMs were more robust in their responses than to misspellings. The term “Kanna operation” was well understood by most models, and Hinglish queries were responded to in a manner that mirrored the GT answer. However, less common terms were not understood when mixed with English. The query with the word “Karwat” as mentioned in 3.1 received responses ranging from “you can start cooking after 1 week” to “I’m sorry, but I cannot provide an answer to your question. The information you are seeking is not relevant to cataract surgery or the guidelines for post-operative care” to “be careful when children get near you”. Most of the evaluated LLMs understood the use of “sugar” in reference to diabetes, as well as sentences following different syntax than would be common in native English.

The responses for culturally-relevant questions varied greatly between evaluated LLMs. For example, to the question on appropriateness of chapati and puri on the day of surgery, some LLMs approved, saying “Yes, he can take chapati, Puri etc on the day of cataract surgery” while others were against this, saying “You should have a light meal before the surgery. Avoid heavy or oily foods like chapati and Puri on the day of your cataract surgery.”
surgery. It’s best to stick to easily digestible foods. If you have any specific dietary concerns, please discuss them with your healthcare team. Questions relating to returning to a “native place” were met with refusals by around half of the evaluated LLMs.

Questions that were specific to the patient’s condition were also responded to in a diverse manner by the evaluated LLMs. For example, the query “Can aztolet20 (atorvastatin and clopidogrel) tablet be taken post surgery” had the GT response “I do not know the answer to your question. If this needs to be answered by a doctor, please schedule a consultation” as there was no mention of this medication in the KB. However, some LLMs approved its use, responding “Yes, you can take the aztolet20 (atorvastatin and clopidogrel) tablet post cataract surgery. However, it is important to follow your doctor’s prescription and instructions” while others responded with mentions of medication that was referred to in the KB, “If you are referring to the IMOL tablet, it is usually taken when you experience pain. However, for other medications, please share the name so I can provide a more accurate answer. Always remember to follow your doctor’s prescription.” Around half refused to answer the question, mirroring the GT.

# 5 Discussion

In this study, we evaluated 24 models on healthcare-related queries in the RAG setting. Our findings revealed many insights which we share below:

# Difference in model scores

We find that the models that we evaluate vary widely in their scores. This indicates that not all models are suitable for use in the healthcare setting, and we find that some models perform worse than expected. For example, GPT-4O and META-LLAMA-3.1-70B-INSTRUCT perform worse than smaller models on this task.

# English vs. Multilingual Queries

Although the number of non-English queries is small, we find that some Indic models perform better on English queries than non-English queries. We also observe that the Factual Correctness score is lower for non-English queries than English queries on average, indicating that models find it difficult to answer non-English queries accurately. This may be due to the cultural and linguistic nuances present in our queries.

# Multilingual vs. Indic models

We evaluate several models that are specifically fine-tuned on Indic languages and on Indic data and observe that they do not always perform well on non-English queries. This could be because several instruction tuned models are tuned on synthetic instruction data which is usually a translation of English instruction data. A notable exception is the AYA-23-35B model, that contains manually created instruction tuning data for different languages and performs well for Hindi. Additionally, several multilingual instruction tuning datasets have short instructions, which may not be suitable for complex RAG settings, which typically have longer prompts and large chunks of data.

# Human vs. LLM-based evaluation

We conduct human evaluation on a subset of models and data points and observe strong alignment with the LLM evaluator overall, especially regarding the final ranking of the models. However, for certain models like MISTRAL-LARGE-INSTRUCT-2407 (for Telugu) and META-LLAMA-3.1-70B-INSTRUCT (for other languages), the agreement is low. It is important to note that we use LLM-evaluators both with and without references, and assess human agreement for SEMANTIC SIMILARITY which uses ground truth references. This suggests that LLM-evaluators should be used cautiously in a multilingual context, and we plan to broaden human evaluation to include more metrics in future work.

# Evaluation in controlled settings with uncontaminated datasets

We evaluate 24 models in an identical setting, leading to a fair comparison between models. Our dataset is curated based on questions from users of an application and is not contaminated in the training dataset of any of the models we evaluate, lending credibility to the results and insights we gather.

# Locally-grounded, non-translated datasets

Our dataset includes various instances of code-switching, Indian English colloquialisms, and culturally specific questions which cannot be obtained by translating datasets, particularly with automated translations. While models were able to handle code-switching to a certain extent, responses varied greatly to culturally-relevant questions. This underscores the importance of collecting datasets from target populations while building models or systems for real-world use.

# 6 Limitations

Our work is subject to several limitations.

- Because our dataset is derived from actual users of a healthcare bot, we couldn’t regulate the ratio of English to non-English queries. Consequently, the volume of non-English queries in our dataset is significantly lower than that of English queries, meaning the results on non-English queries should not be considered definitive. Similarly, since the HEALTHBOT is available only in four Indian languages, we also could not evaluate on languages beyond these. The scope of our HEALTHBOT setting is currently confined to queries from patients at one hospital in India, resulting in less varied data. We intend to expand this study as HEALTHBOT extends its reach to other parts of the country.
- While we evaluated numerous models in this work, some were excluded from this study for various reasons, such as ease of access. We aim to incorporate more models in future research.
- Research has indicated that LLM-based evaluators tend to prefer their own responses. In our evaluations, we use GPT-4O, and there may be a bias leading to higher scores for the GPT-4O model and other models within the GPT family. Although not investigated in prior research, it is also conceivable that models fine-tuned with synthetic data generated by GPT-4O might receive elevated scores. We urge readers to keep these in mind while interpreting the scores. In future work, we plan to use multiple LLM-evaluators to obtain more robust results.
- Finally, our human evaluation was limited to a subset of models and data, and a single metric due to time and budget constraints. In future work, we plan to incorporate more human evaluation, as well as qualitative analysis of the results.

# 7 Ethical Considerations

We use the framework by Bender and Friedman (2018) to discuss the ethical considerations for our work.

# 8 Acknowledgements

We thank Aditya Yadavalli, Vivek Seshadri, the Operations team and Annotators from KARYA for the streamlined annotation process. We also extend our gratitude to Bhuvan Sachdeva for helping us with the HEALTHBOT deployment, data collection and organization process.
