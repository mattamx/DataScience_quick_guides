# Sampling and point estimates

Population vs. sample

The *population* is the complete dataset
- Doesn't have to refer to people
- Typically, don't know what the whole population is

The *sample* is the subset of data you calculate on

Coffee rating dataset
| total_cups_points | variety | country_of_origin | aroma | flavor | aftertaste | body | balance
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----
| 90.58 | NA | Ethiopia | 8.67 | 8.83 | 8.67 | 8.50 | 8.42
| 89.92 | Other | Ethiopia | 8.75 | 8.67 | 8.50 | 8.42 | 8.42
| ... | ... | ... | ... | ... | ... | ... | ...
| 73.75 | NA | Vietname | 6.75 | 6.67 | 6.5 | 6.92 | 6.83

- Each row represents 1 coffee
- 1338 rows
- Treated as the population

Points vs. flavor: population
```python
pts_vs_flavor_pop = coffee_ratings[['total_cups_points','flavor']]

# 10 row sample
pts_vs_flavor_samp = pts_vs_flavor_pop.sample(n=10)
```

Python sampling for Series

- Use `.sample()` for `pandas` DataFrames and Series
```python
cup_points_samp = coffee_ratings['total_cup_points'].sample(n=10)
```

Population parameters & point estimates

A *population parameter* is a calculation made on the population dataset
```python
import numpy as np
np.mean(pts_vs_flavor_pop['total_cup_points'])
```
A *point estimate* or *sample statistic* is a calculation made on the sample dataset
```python
np.mean(cup_points_samp)
```
Point estimates with pandas
```python
pts_vs_flavor_pop['flavor'].mean() # 7.526

pts_vs_flavor_samp['flavor'].mean() # 7.485
```

# Convenience sampling

Convenience sampling coffee ratings
```python
coffee_ratings['total_cup_points'].mean() # 82.151

coffee_ratings_first10 = coffee_ratings.head(10)

coffee_ratings_first10['total_cup_points'].mean() # 89.1
```

Visualizing selection bias
```python
import matplotlib.pyplot as plt
import numpy as np

coffee_ratings['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()

coffee_ratings_first10['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()
```
**Distribution of a population and of a convenience sample**

<kbd> <img width="742" alt="Screenshot 2024-01-29 at 4 41 45 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7d4155a5-1f1a-4504-83f3-0afbbecd83d3"> </kbd>

Visualizing selection bias for a random sample
```python
coffee_sample = coffee_ratings.sample(n=10)
coffee_sample['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()
```

**Distribution of a population and of a convenience sample**

<kbd> <img width="744" alt="Screenshot 2024-01-29 at 4 43 04 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/68e2f663-7f1f-4987-9b84-1e5fbe63d465"> </kbd>

# Pseudo-random number generation

- Pseudo-random number generation is **cheap** and **fast**
- Next "random" number calcualted from previous "random" number
- The first "random" number calculated from a *seed*
- The same seed value yields the same random numbers

Example
```python
seed = 1
calc_next_random(seed) # 3

calc_next_random(3) # 2

calc_next_random(2) # 6
```

**Random number generating functions**

- Prepend with `numpy.random`, such as `numpy.random.beta()`

| function | distribution | function | distribution
| ----- | ----- | ----- | -----
| .beta | Beta | .hypergeometric | Hypergeometric
| .binomial | Binomial | .lognormal | Lognormal
| .chisquare | Chi-squared | .negative_binomial | Negative binomial
| .exponential | Exponential | .normal | Normal
| .f | F | .poisson | Poisson
| .gamma | Gamma | .standard_t | t
| .geometric | Geometric | .uniform | Uniform


Visualizing random numbers
```python
randoms = np.random.beta(a=2, b=2, size=5000)

plt.hist(randoms, bins=np.arange(0,1,0.05)
plt.show()
```
<kbd> <img width="332" alt="Screenshot 2024-01-29 at 4 49 33 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/b0f9ef97-f836-4d98-b55c-2b74cef837a8">
</kbd>

Random numbers seeds
```python
np.random.seed(20000229)

np.random.normal(loc=2, scale=1.5, size=2)
np.random.normal(loc=2, scale=1.5, size=2)

# using a different seed
np.random.seed(20041004)

np.random.normal(loc=2, scale=1.5, size=2)
np.random.normal(loc=2, scale=1.5, size=2)
```

**Seed 20000229**

<kbd> <img width="277" alt="Screenshot 2024-01-29 at 4 52 13 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/1917a708-7d9b-4a3e-9034-39840a15c237"></kbd>
<kbd> <img width="260" alt="Screenshot 2024-01-29 at 4 52 21 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/06f13c4e-f37c-4aae-b0ad-f71d23c9a1e6"></kbd>

**Seed 20041004**

<kbd><img width="256" alt="Screenshot 2024-01-29 at 4 53 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/b7f7eeb2-b251-4930-b6cc-90bc88d193a7"></kbd>
<kbd><img width="267" alt="Screenshot 2024-01-29 at 4 53 27 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3a2b42fe-c12f-4ce8-91e5-e3a0038c84e5"></kbd>

# Simple random and systemic sampling

Simple randomg sampling with pandas
```python
coffee_ratings_sample(n=5, random_state=19000113)
```
Systemic sampling - defining the interval
```python
sample_size = 5
pop_size = len(coffee_ratings) # 1338

interval = pop_size // sample_size # 267
```
Systemic sampling = selecting the rows
```python
coffee_ratings.iloc[::interval]
```
The trouble with systemic sampling

```python
coffee_ratings_with_id = coffee_ratings.reset_index()
coffee_ratings_with_id.plot(x='index', y='aftertaste', kind='scatter')
plt.show()
```
<kbd> <img width="320" alt="Screenshot 2024-01-29 at 4 57 21 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/ab5402a8-06f8-4dcd-a8ff-991d6585e5f3">
</kbd>
Systemic sampling is only safe if we don't see a patter in this scatter plot

Making systemic sampling safe
```python
shuffled = coffee_ratings.sample(frac=1)
shuffled = shuffled.reset_index(drop=True).reset_index()
shuffled.plot(x='index', y='aftertaste', kind='scatter')
plt.show()
```

<kbd> <img width="292" alt="Screenshot 2024-01-29 at 4 59 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d97e404f-91f7-437a-b21b-24de46d995e9"> </kbd>
Shuffling rows + systematic sampling is the same as simple random sampling

# Stratified and weighted random sampling

Coffees by country
```python
top_counts = coffee_ratings['country_of_origin'].value_counts()
top_counts.head(6)
```
Filtering for 6 countries
```python
top_counted_countries = ["Mexico", "Colombia", "Guatemala",
                        "Brazil", "Taiwan", "United States (Hawaii)"]

top_counted_subset = coffee_ratings['country_of_origin'].isin(top_counted_countries)

coffee_ratings_top = coffee_ratings[top_counted_subset]
```

Counts of a simple random sample
```python
coffee_ratings_samp = coffee_ratings.sample(frac=1, random_state=2021)
coffee_ratings_samp['country_of_origin'].value_counts(normalize=True)
```

**Comparing proportions**

<kbd> <img width="767" alt="Screenshot 2024-01-29 at 5 03 03 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/6d62393e-d567-4737-97fb-12986c0d804d">
</kbd>

Proportional stratified sampling
```python
coffee_ratings_strat = coffee_ratings_top.groupby('country_of_origin')\
    .sample(frac=0.1, random_state=2021)

coffee_ratings_strat['country_of_origin'].value_counts(normalize=True)
```
<kbd> <img width="377" alt="Screenshot 2024-01-29 at 5 04 46 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c9280fc9-38ef-4b42-9b26-e67f455239c7">
</kbd>

Equal counts stratified sampling
```python
coffee_ratings_eq = coffee_ratings_top.groupby('country_of_origin')\
    .sample(n=15, random_state=2021)

coffee_ratings_eq['country_of_origin'].value_counts(normalize=True)
```
<kbd><img width="366" alt="Screenshot 2024-01-29 at 5 05 48 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/bedd284b-7cc1-407d-bb4d-eef05420847e">
</kbd>

Weighted random sampling

- Specify weights to adjust the relative probability of a row being sampled
```python
import numpy as np

coffee_ratings_weight = coffee_ratings_top
condition = coffee_ratings_weight['country_of_origin'] == 'Taiwan'

coffee_ratings_weight['weight'] = np.where(condition, 1, 2)

coffee_ratings_weight = coffee_ratings_weight.sample(frac=0.1, weights='weight')
```

Weighted random sampling results

10% weighted sample
```python
coffee_ratings_weight['country_of_origin'].value_counts(normalize=True)
```
<kbd><img width="363" alt="Screenshot 2024-01-29 at 5 08 49 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/dbd4f75b-88e6-4978-82b2-cd048b42de76">
</kbd>


# Cluster sampling
```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
