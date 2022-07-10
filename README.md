## docuscospacy

The **docuscospacy** package contains a set of functions to facilitate the processing of corpora using:

* [en\_docusco_spacy](https://huggingface.co/browndw/en_docusco_spacy) -- a spaCy model trained on the [CLAWS7](https://ucrel.lancs.ac.uk/claws7tags.html) tagset and DocuScope; and
* [tmtoolkit](https://tmtoolkit.readthedocs.io/en/latest/) --  a set of tools for text mining and topic modeling

### What is DocuScope?

DocuScope is a dictionary-based tagger, developed by [David Kaufer and Suguru Ishizaki](https://www.igi-global.com/chapter/computer-aided-rhetorical-analysis/61054) at Carnegie Mellon University. You can find an early version of the dictionary [here](https://github.com/docuscope/DocuScope-Dictionary-June-26-2012).

The tagger organizes words and phrases into 38 rhetorically oriented categories. (You can find descriptions of the categories below.) The spaCy model was trained on data tagged with the 2020 version of the dictionary.

### Why use the spaCy model?

DocuScope has been used in a wide variety of studies. It has proven particularly effective in modelling variation (in genre, register, and style) in a variety of historical, literary, educational, and political texts (Brown and Laudenbach 2021, Hope and Witmore 2004, Marcellino 2014, Parry-Giles and Kaufer 2017, Taguchi et al. 2017, Tootalian 2017, Zhao and Kaufer 2013).

The spaCy model makes some of that explanatory power more readily available to researchers, students, and NLP professionals.

Additionally, the model was trained on the CLAWS7 part-of-speech tagset. This is useful for anyone who wants to compare results to any of [the BYU family of corpora](https://www.english-corpora.org/), which uses the same system.

Note that there is also [a more accurate model trained on BERT](https://huggingface.co/browndw/docusco-bert). However, that model requires more expertise to implement, particularly on longer texts.


### Model output

Many DocuScope tokens are made up of multiple words. Thus, the model was trianed using a NER pipeline and a typical IOB scheme.

DocuScope tags can, therefore, be accessing using any of the **ent** attributes and the CLAWS7 tags using the **tag** attributes.

For example, tokenizing the sentence:

> Jaws is a shrewd cinematic equation which not only gives you one or two very nasty turns when you least expect them but, possibly more important, knows when to make you think another is coming without actually providing it.

would produce:

| |text|tag\_|ent\_|ent\_type_ |
|---|---|---|---|--- |
| 0|Jaws|NN1|B|Character |
| 1|is|VBZ|B|InformationStates |
| 2|a|AT1|I|InformationStates |
| 3|shrewd|JJ|B|Strategic |
| 4|cinematic|JJ|B|PublicTerms |
| 5|equation|NN1|B|AcademicTerms |
| 6|which|DDQ|B|SyntacticComplexity |
| 7|not|XX|B|ForceStressed |
| 8|only|RR|I|ForceStressed |
| 9|gives|VVZ|B|Interactive |
| 10|you|PPY|I|Interactive |
| 11|one|MC1|O| |
| 12|or|CC|B|MetadiscourseCohesive |
| 13|two|MC|B|InformationExposition |
| 14|very|RG|B|ConfidenceHigh |
| 15|nasty|JJ|B|Negative |
| 16|turns|NN2|O| |
| 17|when|RRQ|B|Narrative |
| 18|you|PPY|I|Narrative |
| 19|least|RRT|B|InformationExposition |
| 20|expect|VV0|B|Future |
| 21|them|PPHO2|B|Narrative |
| 22|but|CCB|B|MetadiscourseCohesive |
| 23|,|Y|B|Contingent |
| 24|possibly|RR|I|Contingent |
| 25|more|RGR|B|InformationExposition |
| 26|important|JJ|I|InformationExposition |
| 27|,|Y|O| |
| 28|knows|VVZ|B|ConfidenceHigh |
| 29|when|RRQ|I|ConfidenceHigh |
| 30|to|TO|O| |
| 31|make|VVI|B|Interactive |
| 32|you|PPY|I|Interactive |
| 33|think|VVI|B|Character |
| 34|another|DD1|B|MetadiscourseCohesive |
| 35|is|VBZ|B|InformationStates |
| 36|coming|VVG|O| |
| 37|without|IW|O| |
| 38|actually|RR|B|ForceStressed |
| 39|providing|VVG|B|Facilitate |
| 40|it|PPH1|O| |
| 41|.|Y|O| |

## Intallation

To intall:

```python
pip intall docuscospacy
```

## Processing a corpus

Preparing text data for analysis requires:

1. Downloading the [en\_docusco_spacy](https://huggingface.co/browndw/en_docusco_spacy) model from the huggingface repository,
2. Loading a spaCy instance from the model,
3. Preparing, loading, and tokenizing a corpus using tmtoolkit's functionalities, and
4. Converting the corpus into a list of nltk-like tuples.

### Loading an instance

Load the model like any spaCy model:

```python
import spacy
from docuscospacy import corpus_analysis
```

```python
nlp = spacy.load('en_docusco_spacy')
```


### Preparing, loading, and tokenizing a corpus

To ensure accurate tagging, pre-processing should be minimal, as DocuScope is sensitive to case and to surrounding punctuation.

However, third-person possessive *its* should be split pior to taagging. It is also useful to remove carriage returns, tabs, etc.

This can be accomplished with a simple function passed to a tmtoolkit corpus function:

```python
from tmtoolkit.corpus import Corpus
```

```python
def pre_process(txt):
    txt = re.sub(r'\bits\b', 'it s', txt)
    txt = re.sub(r'\bIts\b', 'It s', txt)
    txt = " ".join(txt.split())
    return(txt)
```

```python
corp = Corpus.from_folder('my_corpus', spacy_instance=nlp, raw_preproc=[pre_process], spacy_token_attrs=['tag', 'ent_iob', 'ent_type', 'is_punct'])
```

### Converting the corpus

To take advange of the docuscospacy functions, the corpus needs to be converted:

```python
tp = convert_corpus(corp)
```



| Category (Cluster)|Description|Examples |
|---|---|---|
| Academic Terms|Abstract, rare, specialized, or disciplinary-specific terms that are indicative of informationally dense writing|*market price*, *storage capacity*, *regulatory*, *distribution* |
| Academic Writing Moves|Phrases and terms that indicate academic writing moves, which are common in research genres and are derived from the work of Swales (1981) and Cotos et al. (2015, 2017)|*in the first section*, *the problem is that*, *payment methodology*, *point of contention* |
| Character|References multiple dimensions of a character or human being as a social agent, both individual and collective|*Pauline*, *her*, *personnel*, *representatives* |
| Citation|Language that indicates the attribution of information to, or citation of, another source.|*according to*, *is proposing that*, *quotes from* |
| Citation Authorized|Referencing the citation of another source that is represented as true and not arguable|*confirm that*, *provide evidence*, *common sense* |
| Citation Hedged|Referencing the citation of another source that is presented as arguable|*suggest that*, *just one opinion* |
| Confidence Hedged|Referencing language that presents a claim as uncertain|*tends to get*, *maybe*, *it seems that* |
| Confidence High|Referencing language that presents a claim with certainty|*most likely*, *ensure that*, *know that*, *obviously* |
| Confidence Low|Referencing language that presents a claim as extremely unlikely|*unlikely*, *out of the question*, *impossible* |
| Contingent|Referencing contingency, typically contingency in the world, rather than contingency in one's knowledge|*subject to*, *if possible*, *just in case*, *hypothetically* |
| Description|Language that evokes sights, sounds, smells, touches and tastes, as well as scenes and objects|*stay quiet*, *gas-fired*, *solar panels*, *soft*, *on my desk* |
| Facilitate|Language that enables or directs one through specific tasks and actions|*let me*, *worth a try*, *I would suggest* |
| First Person|This cluster captures first person.|*I*, *as soon as I*, *we have been* |
| Force Stressed|Language that is forceful and stressed, often using emphatics, comparative forms, or superlative forms|*really good*, *the sooner the better*, *necessary* |
| Future|Referencing future actions, states, or desires|*will be*, *hope to*, *expected changes* |
| Information Change|Referencing changes of information, particularly changes that are more neutral|*changes*, *revised*, *growth*, *modification to* |
| Information Change Negative|Referencing negative change|*going downhill*, *slow erosion*, *get worse* |
| Information Change Positive|Referencing positive change|*improving*, *accrued interest*, *boost morale* |
| Information Exposition|Information in the form of expository devices, or language that describes or explains, frequently in regards to quantities and comparisons|*final amount*, *several*, *three*, *compare*, *80%* |
| Information Place|Language designating places|*the city*, *surrounding areas*, *Houston*, *home* |
| Information Report Verbs|Informational verbs and verb phrases of reporting|*report*, *posted*, *release*, *point out* |
| Information States|Referencing information states, or states of being|*is*, *are*, *existing*, *been* |
| Information Topics|Referencing topics, usually nominal subjects or objects, that indicate the “aboutness” of a text|*time*, *money*, *stock price*, *phone interview* |
| Inquiry|Referencing inquiry, or language that points to some kind of inquiry or investigation|*find out*, *let me know if you have any questions*, *wondering if* |
| Interactive|Addresses from the author to the reader or from persons in the text to other persons. The address comes in the language of everyday conversation, colloquy, exchange, questions, attention-getters, feedback, interactive genre markers, and the use of the second person.|*can you*, *thank you for*, *please see*, *sounds good to me* |
| Metadiscourse Cohesive|The use of words to build cohesive markers that help the reader navigate the text and signal linkages in the text, which are often additive or contrastive|*or*, *but*, *also*, *on the other hand*, *notwithstanding*, *that being said* |
| Metadiscourse Interactive|The use of words to build cohesive markers that interact with the reader|*I agree*, *let’s talk*, *by the way* |
| Narrative|Language that involves people, description, and events extending in time|*today*, *tomorrow*, *during the*, *this weekend* |
| Negative|Referencing dimensions of negativity, including negative acts, emotions, relations, and values|*does not*, *sorry for*, *problems*, *confusion* |
| Positive|Referencing dimensions of positivity, including actions, emotions, relations, and values|*thanks*, *approval*, *agreement*, *looks good* |
| Public Terms|Referencing public terms, concepts from public language, media, the language of authority, institutions, and responsibility|*discussion*, *amendment*, *corporation*, *authority*, *settlement* |
| Reasoning|Language that has a reasoning focus, supporting inferences about cause, consequence, generalization, concession, and linear inference either from premise to conclusion or conclusion to premise|*because*, *therefore*, *analysis*, *even if*, *as a result*, *indicating that* |
| Responsibility|Referencing the language of responsibility|*supposed to*, *requirements*, *obligations* |
| Strategic|This dimension is active when the text structures strategies activism, advantage-seeking, game-playing cognition, plans, and goal-seeking.|*plan*, *trying to*, *strategy*, *decision*, *coordinate*, *look at the* |
| Syntactic Complexity|The features in this category are often what are called “function words,” like determiners and prepositions.|*the*, *to*, *for*, *in*, *a lot of* |
| Uncertainty|References uncertainty, when confidence levels are unknown|*kind of*, *I have no idea*, *for some reason* |
| Updates|References updates that anticipate someone searching for information and receiving it|*already*, *a new*, *now that*, *here are some* |

### BibTeX entry and citation info
```
@incollection{ishizaki2012computer,
  title    = {Computer-aided rhetorical analysis},
  author   = {Ishizaki, Suguru and Kaufer, David},
  booktitle= {Applied natural language processing: Identification, investigation and resolution},
  pages    = {276--296},
  year     = {2012},
  publisher= {IGI Global},
  url      = {https://www.igi-global.com/chapter/content/61054}
}
```
```
@article{brown2021stylistic,
  title={Stylistic variation in email},
  author={Brown, David West and Laudenbach, Michael},
  journal={Register Studies},
  year={2021},
  publisher={John Benjamins Publishing Company Amsterdam/Philadelphia}
}
```
```
@article{hope2004very,
  title={The very large textual object: a prosthetic reading of Shakespeare},
  author={Hope, Jonathan and Witmore, Michael},
  journal={Early Modern Literary Studies},
  volume={9},
  number={3},
  pages={1--36},
  year={2004},
  publisher={Matthew Steggle}
}
```
```
@article{marcellino2014talk,
  title={Talk like a Marine: USMC linguistic acculturation and civil--military argument},
  author={Marcellino, William M},
  journal={Discourse Studies},
  volume={16},
  number={3},
  pages={385--405},
  year={2014},
  publisher={Sage Publications Sage UK: London, England}
}
```
```
@article{kaufer2017hillary,
  title={Hillary Clinton’s presidential campaign memoirs: A study in contrasting identities},
  author={Kaufer, David S and Parry-Giles, Shawn J},
  journal={Quarterly Journal of Speech},
  volume={103},
  number={1-2},
  pages={7--32},
  year={2017},
  publisher={Taylor \& Francis}
}
```
```
@article{taguchi2017corpus,
  title={A corpus linguistics analysis of on-line peer commentary},
  author={Taguchi, Naoko and Kaufer, David and G{\'o}mez-Laich, Pia Maria and Zhao, Helen},
  journal={Pragmatics and language learning},
  volume={14},
  pages={357--170},
  year={2017}
}
```
```
@article{tootalian2017corrupt,
  title={“To Corrupt a Man in the Midst of a Verse”: Ben Jonson and the Prose of the World},
  author={Tootalian, Jacob},
  journal={Ben Jonson Journal},
  volume={24},
  number={1},
  pages={46--72},
  year={2017},
  publisher={Edinburgh University Press 22 George Square, Edinburgh EH8 9LF UK}
}
```
```
@article{zhao2013docuscope,
  title={DocuScope for genre analysis: Potential for assessing pragmatic functions},
  author={Zhao, Helen and Kaufer, David},
  journal={Technology in interlanguage pragmatics research and teaching},
  pages={235--260},
  year={2013},
  publisher={John Benjamins Amsterdam}
}
```
