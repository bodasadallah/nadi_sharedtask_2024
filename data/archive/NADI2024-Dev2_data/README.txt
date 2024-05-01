================================================================
                    NADI 2024: The Fifth Nuanced Arabic Dialect Identification Shared Task
=================================================================

Releasing of Second Development Set (Dev2)
==========================================
Note that Dev2 is the Development set used for Dev phase on CodaLab competitions.
==========================================

For subtask 1, we provide a development set as a tsv file:
* ./subtask1/NADI2024_subtask1_dev2.tsv

- The (sentence) column contains the content of tweet.
- The columns (Algeria, Egypt, Jordan, Palestine, Sudan, Syria, Tunisia, Yemen) contain the labels for the dialects of the sentences. The labels are binary, "y" if the sentence is valid in a dialect and "n" if it is not.

--------------------------------------------------
For subtask 2, we provide a development set as a tsv file:
* ./subtask2/NADI2024_subtask2_dev2.tsv

- The (sentence) column contains the content of tweet.
- The (ALDi) column contains the average human-annotated ALDi score for the sentence. The score is a float in the range [0, 1], where 0 means MSA, and 1 means high divergence from MSA.

--------------------------------------------------
Subtask 3 (Open track): Dialect to MSA translation:
We provide a Dev set:
    ./subtask_3/NADI2024_subtask3_dev.tsv

For DEV files of Subtask 3, we provide the following information:
- The first column (dialect_id) contains a name of the source country-level dialect. 
- The second column (source_dialect) contains a source dialect text.
- The third column (target_msa) contains a translated target text in MSA.

=================================================================
subtask1/NADI2024-ST1-Scorer.py and Submission samples
=================================================================
./Subtask_1/NADI2024-ST1-Scorer.py is a python script for ** Subtask 1 ** that takes in two text files containing true labels and predicted labels and will output macro-averaged accuracy,F1 score, precision, and recall. Note that the official metric for subtask 1 is the macro F1 score across the included N dialects.

*** Each line of your prediction file should have 18 boolean values separated by commas, where each value represents the validity of the sentence (1) or invalidilty (0) in the following order: Algeria, Bahrain, Egypt, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Palestine, Qatar, Saudi Arabia, Sudan, Syria, Tunisia, UAE, Yemen.

*** If the gold file has labels for "m" dialects, only the predictions of these dialects will be used to evaluate the system. However, your system should still make predictions for all 18 dialects.

-------------------------------------
Usage of the scorer:

In the dataset directory, there are example gold and prediction files. If they are used with the scorer, they should produce the following results:

python3 NADI2024-ST1-Scorer.py NADI2024_subtask1_dev2_gold.txt UBC_subtask1_dev_1.txt

OVERALL SCORES:
MACRO AVERAGE PRECISION SCORE: 38.14 %
MACRO AVERAGE RECALL SCORE: 50.33 %
MACRO AVERAGE F1 SCORE: 42.47 %
MACRO AVERAGE ACCURACY: 50.83 %

=================================================================
subtask_2/NADI2024-ST2-Scorer.py and Submission samples
=================================================================
./Subtask_2/NADI2024-ST2-Scorer.py is a python script for ** Subtask 2** that takes in two text files containing GOLD ALDi values and predicted ALDi values and will output the RMSE between these two lists of scores.

-------------------------------------
Usage of the scorer:

We provide a sample of gold file and a submission sample file for Subtask 2. 
Please make sure to have evaluate library installed.

python3 NADI2024-ST2-Scorer.py NADI2024_subtask2_dev2_gold.txt UBC_subtask2_dev_1.txt

RMSE: 0.35579

=================
Sharing NADI Data
=================
Since we are sharing the actual tweets, and as an additional measure to protect Twitter user privacy, we ask that participants not share the distributed data outside their labs nor publish these tweets publicly. Any requests for use of the data outside the shared task can be directed to organizers and will only be accommodated after the shared task submission deadline.

=================================================================
subtask_3/NADI2024-ST3-Scorer.py and Submission samples
=================================================================
./Subtask_3/NADI2024-ST3-Scorer.py is a python script for ** Subtask 3 ** that takes in two text files containing GOLD target text and generated text and will output BLEU scores for represented dialect and overall score. 

We provide a sample of gold file and a submission sample file for Subtask 3. 
Please make sure to have evaluate library installed.

`./subtask_3/NADI2024_subtask3_dev_GOLD.txt' is the gold target text file of Dev set of subtask 3. 

The file `./subtask_3/UBC_subtask3_dev_1.zip' is the zip file of my first submission.
This zip file contains only one txt file: `UBC_subtask3_dev_1.txt'. 
Unzipping `UBC_subtask3_dev_1.zip', you can get `UBC_subtask3_dev_1.txt' where each line is a generated translation. 

`NADI2024_subtask3_dev_GOLD.txt' and `UBC_subtask3_dev_1.txt' can be used with the NADI2024-ST3-Scorer.py.

-------------------------------------
Usage of the scorer:

In the dataset directory, there are example gold and prediction files. If they are used with the scorer, they should produce the following results:

python3 ./subtask3/NADI2024-ST3-Scorer.py NADI2024_subtask3_dev_GOLD.txt UBC_subtask3_dev_1.txt

Scores:
{'Overall': 19.799181764464386, 'Egyptain': 13.248322373167568, 'Emirati': 0.0, 'Jordanian': 59.76006721116457, 'Palestinian': 6.583638322992827}


-------------------------------------
IMPORTANT: Participants are NOT allowed to use **NADI2024_Subtask1_DEV.tsv**, **NADI2024_Subtask2_DEV.tsv** or **NADI2024_Subtask3_DEV.tsv** for training purposes. Participants must report the performance of their best system on both DEV *and* TEST sets in their Shared Task system description paper.

-------------------------------------
For Subtask 1:
IMPORTANT: Participants can only use the official TRAIN sets provided by NADI-2024.
Participants are NOT allowed to use any additional tweets, nor are they allowed to use outside information.
Specifically -- participants should not use meta data from Twitter about the users or the tweets, e.g., geo-location data.
External manually labelled data sets are *NOT* allowed.

-------------------------------------
For Subtask 2:
IMPORTANT: Participants are free to select their own or develop new training datasets for training. 
Participants are allowed to use public datasets such as ALDI-AOC (Keleg et al, 2023b) which is a transformation of the AOC dataset (Zaidan and Callison-Burch, 2014), in addition to raw MSA and dialectal Arabic corpora.

IMPORTANT: Participants should agree to share the full used training dataset they create with the community. We will collect these datasets during submission time and facilitate their distribution from direct download links. 

-------------------------------------
For Subtask 3:
IMPORTANT: Participants are free to select their own or develop new training datasets for training. 
Participants can manually translate Mono-DA and/or Mono-MSA for that purpose, but they can also choose to acquire and translate any other datasets into any of the four dialects listed.

IMPORTANT: Participants should agree to share the full used training dataset they create with the community. We will collect these datasets during submission time and facilitate their distribution from direct download links. 

-------------------------------------
NADI2024-Twitter-Corpus-License.txt contains the license for using
this data.


=================================================================
References
=================================================================
@inproceedings{keleg-etal-2023-aldi,
    title = "{ALD}i: Quantifying the {A}rabic Level of Dialectness of Text",
    author = "Keleg, Amr  and
      Goldwater, Sharon  and
      Magdy, Walid",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.655",
    doi = "10.18653/v1/2023.emnlp-main.655",
    pages = "10597--10611"
}

@inproceedings{abdul-mageed-etal-2020-nadi,
    title = "{NADI} 2020: The First Nuanced {A}rabic Dialect Identification Shared Task",
    author = "Abdul-Mageed, Muhammad  and
      Zhang, Chiyu  and
      Bouamor, Houda  and
      Habash, Nizar",
    booktitle = "Proceedings of the Fifth Arabic Natural Language Processing Workshop",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.wanlp-1.9",
    pages = "97--110",
    
}

@inproceedings{abdul-mageed-etal-2021-nadi,
    title = "{NADI} 2021: The Second Nuanced {A}rabic Dialect Identification Shared Task",
    author = "Abdul-Mageed, Muhammad  and
      Zhang, Chiyu  and
      Elmadany, AbdelRahim  and
      Bouamor, Houda  and
      Habash, Nizar",
    booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
    month = apr,
    year = "2021",
    address = "Kyiv, Ukraine (Virtual)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wanlp-1.28",
    pages = "244--259",
    
}

@inproceedings{abdul-mageed-etal-2022-nadi,
    title = "{NADI} 2022: The Third Nuanced {A}rabic Dialect Identification Shared Task",
    author = "Abdul-Mageed, Muhammad  and
      Zhang, Chiyu  and
      Elmadany, AbdelRahim  and
      Bouamor, Houda  and
      Habash, Nizar",
    booktitle = "Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wanlp-1.9",
    pages = "85--97",
    
}

@inproceedings{abdul-mageed-etal-2023-nadi,
    title = "{NADI} 2023: The Fourth Nuanced {A}rabic Dialect Identification Shared Task",
    author = "Abdul-Mageed, Muhammad  and
      Elmadany, AbdelRahim  and
      Zhang, Chiyu  and
      Nagoudi, El Moatez Billah  and
      Bouamor, Houda  and
      Habash, Nizar",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.62",
    doi = "10.18653/v1/2023.arabicnlp-1.62",
    pages = "600--613",
    
}

@inproceedings{abdul-mageed-etal-2018-tweet,
    title = "You Tweet What You Speak: A City-Level Dataset of {A}rabic Dialects",
    author = "Abdul-Mageed, Muhammad  and
      Alhuzali, Hassan  and
      Elaraby, Mohamed",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Hasida, Koiti  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios  and
      Tokunaga, Takenobu",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1577",
}

@inproceedings{abdul-mageed-etal-2020-toward,
    title = "Toward Micro-Dialect Identification in Diaglossic and Code-Switched Environments",
    author = "Abdul-Mageed, Muhammad  and
      Zhang, Chiyu  and
      Elmadany, AbdelRahim  and
      Ungar, Lyle",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.472",
    doi = "10.18653/v1/2020.emnlp-main.472",
    pages = "5855--5876",
}

@inproceedings{keleg-magdy-2023-arabic,
    title = "{A}rabic Dialect Identification under Scrutiny: Limitations of Single-label Classification",
    author = "Keleg, Amr  and
      Magdy, Walid",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.31",
    doi = "10.18653/v1/2023.arabicnlp-1.31",
    pages = "385--398",
    
}

@article{zaidan-callison-burch-2014-arabic,
    title = "{A}rabic Dialect Identification",
    author = "Zaidan, Omar F.  and
      Callison-Burch, Chris",
    journal = "Computational Linguistics",
    volume = "40",
    number = "1",
    month = mar,
    year = "2014",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/J14-1006",
    doi = "10.1162/COLI_a_00169",
    pages = "171--202",
}

@inproceedings{bouamor-etal-2018-madar,
    title = "The {MADAR} {A}rabic Dialect Corpus and Lexicon",
    author = "Bouamor, Houda  and
      Habash, Nizar  and
      Salameh, Mohammad  and
      Zaghouani, Wajdi  and
      Rambow, Owen  and
      Abdulrahim, Dana  and
      Obeid, Ossama  and
      Khalifa, Salam  and
      Eryani, Fadhl  and
      Erdmann, Alexander  and
      Oflazer, Kemal",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1535",
}