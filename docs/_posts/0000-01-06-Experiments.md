---
layout: slide
title: "Experiments"
---
We originally experimented with the more complex BERT representation of the text. The thinking behind this was that the BERT encodings would be able to capture a better understanding of the text both semantically and contextually. We experimented with many different methods of fine tuning BERT, attempting to fine tune a single classifier layer on to of pooled
However we were unable to produce results near that of claudette, with our best variants of the fine tuned BERT model unable to crack an f1-score of .6