# Text-Simplification-BA
Willkommen im Repository zu der praktischen Implementierung meiner Bachelor-Arbeit mit dem Thema: 
***Textvereinfachung durch Fine-Tuning von Large Language Models. Möglichkeiten zur Aufbereitung von Texten für Kinder unterschiedlicher Alters- und Sprachniveaus***

## Verzeichnisstruktur

```
.
├── automatic_evaluation/     # Automatische Metriken und Ergebnisse (Satzlänge, AoA, WSTF, ...)
├── data/                     # Trainings-, Validation- und Test-Daten, sowie die Skripte zum Crawlen und Erstellen dieser
├── human_evaluation/         # Model-Outputs + Skripts für die Generierung dieser für das Survey, sowie die Wikipedia Ursprungs-Texte
├── lexical_ressources/       # Age of Acquisition und Wortfrequenzdaten aus Kinderbüchern (nicht praktisch verwendet)
└── model/                    # Training, Hyperparameter-Tuning und finale Pipeline
.gitignore
README.md            
requirements.txt              # Kein Anspruch auf Vollständigkeit, aber das meiste sollte enthalten sein
```

## Referenzen

**Age of Acquisition**

Conde, J., Martínez, G., Grandury, M., Arriaga, C., Haro, J., Schroeder, S., Hintz, F., Reviriego, P. and Brysbaert, M. (2026) ‘Updating the German Psycholinguistic Word Toolbox with AI-Generated Estimates of Concreteness, Valence, Arousal, Age of Acquisition, and Familiarity’, Journal of Cognition, 9(1), p. 9. Available at: https://doi.org/10.5334/joc.482.

https://osf.io/ghjd2/overview [besucht am 26.08.2026]

**Wortfrequenz**

Lange, Willi/Okamura, Saburo/Scharloth, Joachim (2016): Datengeleiteter Kernwortschatz Deutsch.  Verfügbar unter: http://www.basic-german.com [besucht am 15.08.2026] (Ranglisten der häufigsten Wörter, Grundwortschatz Kinder [Auswertung Kinderbücher])

**Klexikon**

Die Trainings-Daten stammen aus dem [Klexikon](https://klexikon.zum.de/wiki/Klexikon:%C3%9Cber_uns), einem Online-Lexikon für Kinder von 6-14 Jahren.

**D-Sari**
@misc{sun2021documentleveltextsimplificationdataset,
      title={Document-Level Text Simplification: Dataset, Criteria and Baseline}, 
      author={Renliang Sun and Hanqi Jin and Xiaojun Wan},
      year={2021},
      eprint={2110.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2110.05071}, 
}
GitHub
