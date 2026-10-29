# Demo: item-based collaborative filtering (Stage 8)

Model: ItemCF(k=20, shrink=10, damp=1.0), fitted on train_full (4029 ratings, 309 users, 126 movies). Settings are untuned (Stage 9).

## User 8 (generator taste group: Hollywood sci-fi, used only to choose a demo user)

**Highest-rated movies in their history**

            movie    genre language  rating
Pariyerum Perumal    Drama    Tamil     5.0
      The Martian   Sci-Fi  English     5.0
       Veer-Zaara  Romance    Hindi     4.5
       Knives Out Thriller  English     4.5
           Vikram   Action    Tamil     4.5

**Top-5 recommendations (unseen movies, by predicted rating)**

                movie_title    genre  language  score
                 Magadheera   Action    Telugu  4.743
                    Dhoom 2   Action     Hindi  4.649
Agent Sai Srinivasa Athreya Thriller    Telugu  4.621
                     U Turn Thriller   Kannada  4.575
    Maheshinte Prathikaaram   Comedy Malayalam  4.545

## User 4 (generator taste group: Southern action, used only to choose a demo user)

**Highest-rated movies in their history**

                  movie  genre language  rating
                    RRR Action   Telugu     5.0
Ala Vaikunthapurramuloo Comedy   Telugu     5.0
                   Pink  Drama    Hindi     5.0
                 Kaithi Action    Tamil     5.0
                Dhoom 2 Action    Hindi     5.0

**Top-5 recommendations (unseen movies, by predicted rating)**

            movie_title    genre  language  score
             Magadheera   Action    Telugu  5.000
           Forrest Gump    Drama   English  4.953
                 Barfi!  Romance     Hindi  4.903
Maheshinte Prathikaaram   Comedy Malayalam  4.839
            A Wednesday Thriller     Hindi  4.798

## Movies most similar to "Inception"

       movie_title  genre language  similarity  co_raters
Mad Max: Fury Road Action  English       0.199          3
    Dil Chahta Hai Comedy    Hindi       0.185          4
            Kaithi Action    Tamil       0.180          5
             Queen Comedy    Hindi       0.180          3
          Enthiran Sci-Fi    Tamil       0.172          3

## Movies most similar to "Baahubali"

                movie_title    genre language  similarity  co_raters
Agent Sai Srinivasa Athreya Thriller   Telugu       0.386         12
                    Kantara   Action  Kannada       0.279          4
               The Hangover   Comedy  English       0.270         18
                 Maanagaram Thriller    Tamil       0.235          9
                      Queen   Comedy    Hindi       0.233          8

## Reading the demo honestly
- The module runs end to end: similarities, predictions, top-N lists that exclude seen movies.
- Validation results (docs/experiments.md): with default settings it only marginally improves RMSE over the bias baseline and does not beat the popularity list on precision@10.
- Item similarities rest on few co-raters (typically 2-3 per pair), so neighbours are noisy. Do not read the lists above as proof of taste understanding.