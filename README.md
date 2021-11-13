#### Intro

This repo contains some code for project: "Early and Late Hybrid Fusion Model for Depression Detection"

##### Dataset

We use the Distress Analysis Interview Corpus -Wizard of Oz (DAIC-WOZ) dataset

##### Method

Model structures of the Implemented C-CNN, LSTM models is in `core/models`, they are used to detect depression symptoms using multi-modal data. 

Script in `utils/feature_produce` extracte text features based on POS tags, vocabulary richness, readability measures, and so on