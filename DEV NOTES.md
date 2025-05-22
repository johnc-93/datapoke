## V0.1 Release goals

- Data quality checks only
- CSV only
- Overview of inferred types and missing values
- Option to coerce to new types, and return problem values

TBC on whether to include some kind of visual representation of these - eg. null distribution. The priority for now is to get some working functions, then consider optimal OO layout.

### How to deal with data coercion errors

What are the practical outcomes a user might want?
- If the aim is to inspect drivers, forcing conversions is most likely, may include as null or quarantine depending on the dataset
- If the aim is to understand data quality, understanding errors will be the priority

Two broad categories:
1. Force coerce, with the option to quarantine or null out (option to exclude data will be covered by quarantine)
2. Understand types of conversion errors, perhaps by returning a sample

Perhaps two functions, one to actually convert and the other to test & return issues? Or a function with a primary and supplemental return? The main computation will be an attempted coercion either way, so it would make sense to combine these in some way to avoid repeating costly operations on large datasets. Include a cache option and save to a property?

Ideally we can save a schema once finalised and create an inspection pipeline.



## Long list considerations:

- Consider python version requirements - likely to use cached properties so python 3.8
- Consider pandas version requirements
- instead of just quarantining coercion records, give options for a schema to force some columns and quarantine others - some kind of quarantine mask