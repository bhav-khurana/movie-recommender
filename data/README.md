# MovieLens Dataset

MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.

This data set consists of:

- 100,000 ratings (1-5) from 943 users on 1682 movies.
- Each user has rated at least 20 movies.

`u.data`: The full u data set, 100000 ratings by 943 users on 1682 items.
Each user has rated at least 20 movies. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of `user id | item id | rating | timestamp`. The time stamps are unix seconds since 1/1/1970 UTC

`ua.base` and `ua.test` are 80%/20% splits of the u data into training and test data.
