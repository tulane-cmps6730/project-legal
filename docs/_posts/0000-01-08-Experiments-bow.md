---
layout: slide
title: "Hybrid Model Experiments"
---
Finally, we experimented with a hybrid model, combining the output of the better performing simple bag-of-words model, with the more complex BERT embeddings. We figured this approach would still let us maintain the complex understanding of the sentence that BERT provides, while getting a nudge in the right direction from the bag-of-words trained model. Ultimately this model was unable to outperform the original bag of words in terms of f1, but it notably had better precision than any BERT model and better recall than any bag of words model. 