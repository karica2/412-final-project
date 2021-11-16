# CS 412 Fall 2021 Final project

Group members: Kenan Arica, Rob Schmidt

THE PLAN: **YOUTUBE COMMENT SPAMS** (<https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection>)

## Collected Data

| Artist    | Video ID    | Spam Comments | Typical Comments | Total Comments |
| --------- | ----------- | :-----------: | :--------------: | :------------: |
|`Psy`      | 9bZkp7q19f0 | 175           | 175              | 350            |
|`KatyPerry`| CevxZvSJLk8 | 175           | 175              | 350            |
|`LMFAO`    | KQ6zr6kCPj8 | 236           | 202              | 438            |
|`Eminem`   | uelHwf8o7_U | 245           | 203              | 448            |
|`Shakira`  | pRpeEdMmmQ0 | 174           | 196              | 370            |

## Deliverables

### DEADLINE EOD 18

- [X] Clean the data
  - [X] Sanitize the messages, get rid of the weird line endings
  - [X] Define a set of important variables and produce a new dataset based on that
    - All values are parsed, however the classification is performed on `CONTENT` and regression should be evaluated upon `AUTHOR`

### DEADLINE EOD 19

- [ ] Bag of words implementation
- [ ] Bag of words on the message, regression on the author name

### AFTER 19

- [ ] Look into other ways of classifying the data
