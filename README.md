# Red String Grouper
*R*ecord *E*quivalence *D*iscoverer based on 
*[String Grouper](https://github.com/Bergvca/string_grouper)* 
(Red String Grouper) is a python package that finds similarities between 
rows/records of a table with multiple fields.  
It is an extension of 
*[String Grouper](https://github.com/Bergvca/string_grouper)*, a library that 
makes finding groups of similar strings within one or two lists of strings 
easy � and fast.

# Installation

    pip install red_string_grouper

# Usage
How do we achieve matching for multiple fields and priorities?

Import the function `record_linkage()` from `red_string_grouper` and specify 
the fields of the table over which the comparison will be made.

```python
import pandas as pd 
import numpy as np 
from red_string_grouper import record_linkage

matches = record_linkage(data_frame, fields_2b_matched_fuzzily,
                         fields_2b_matched_exactly=None,
                         hierarchical=True, max_n_matches=None,
                         similarity_dtype=np.float32, force_symmetries=True,
                         n_blocks=None)
```
                   
This is a function that combines similarity-matching results of several fields of a 
DataFrame (`data_frame`) and returns them in another DataFrame (`matches`).

|Parameter |Status |Description|
|:---|:---:|:---|
|`data_frame`| Required | `pandas.DataFrame` of strings which is the table over which the comparisons will be made.|
|`fields_2b_matched_fuzzily`|Required| List of tuples.  Each tuple is a quadruple: <br>(\<***field name***\>, \<***threshold***\>, \<***ngram_size***\>, \<***weight***\>). <br> \<***field name***\> is the name of a field in `data_frame` which is to be matched using a threshold similarity score of \<***threshold***\> and an ngram size of \<***ngram_size***\>. \<***weight***\> is a number that defines the **relative** importance of the field to other fields -- the field's contribution to the mean similarity will be weighted by this number. <br> \<***weighted mean similarity score***\> = (**&Sigma;**<sub>*field*</sub> \<***weight***\><sub>*field*</sub> &times; \<***similarity***\><sub>*field*</sub>) / (**&Sigma;**<sub>*field*</sub>***weight***<sub>*field*</sub>), <br> where **&Sigma;**<sub>*field*</sub> means "sum over fields".|
|`fields_2b_matched_exactly`| Optional| List of tuples.  Each tuple is a pair: <br> (\<***field name***\>, \<***weight***\>).<br> \<***field name***\> is the name of a field in `data_frame` which is to be matched exactly.  \<***weight***\> has the same meaning as in parameter `fields_2b_matched_fuzzily`. Defaults to `None`. |
|`hierarchical`| Optional | `bool`.  Determines if the output DataFrame will have a hierarchical column-structure (`True`) or not (`False`). Defaults to `True`.|
|`max_n_matches`| Optional | `int`. Maximum number of matches allowed per string.  Defaults to the total number of rows.|
|`similarity_dtype`| Optional| `numpy` type.  Either `np.float32` (the default) or `np.float64`.  A value of `np.float32` allows for less memory overhead during computation but less numerical precision, while `np.float64` allows for greater numerical precision but a larger memory overhead.|
|`force_symmetries`| Optional | `bool`. Specifies whether corrections should be made to the results to account for symmetry thus circumventing some errors which result from loss of numerical significance.  Defaults to `True`.|
|`n_blocks` | Optional | `(int, int)`. This parameter is provided to boost performance, if possible, by splitting the dataset into `n_blocks[0]` blocks for the left operand (of the "comparison operator") and into `n_blocks[1]` blocks for the right operand before performing the string-comparisons blockwise. |

# Examples

```python
import pandas as pd 
from red_string_grouper import record_linkage
```

## Prepare the Input Data:
Here's some sample data:

```python
inputfilename = 'data/us-cities-real-estate-sample-zenrows.csv'
df = pd.read_csv(inputfilename, dtype=str)
```

Note that the data has been read into memory as strings (`dtype=str`), since 
`red_string_grouper` is based on string comparison.

The dataset is a table of 10 000 records:

```python
len(df)
```


    10000


Let us examine the data to determine which fields to compare (that is, use only 
columns without *null* or *NaN* data).
At the same time, we will also check how many unique values each field has: 

```python
for field in df.columns:
    if df[field].nunique() > 1 and not df[field].isna().values.any():
        print(f'{field} : {df[field].nunique()}')
```

    zpid : 10000
    id : 10000
    imgSrc : 9940
    detailUrl : 10000
    statusText : 24
    address : 10000
    addressState : 51
    addressZipcode : 6446
    isUndisclosedAddress : 2
    isZillowOwned : 2
    has3DModel : 2
    hasVideo : 2
    isFeaturedListing : 2
    list : 2
    

We may set field `'zpid'` as the index, since it has exactly the same number 
of unique values as the number of rows.  `zpid` will thus be used to identify 
each row.


```python
df.set_index('zpid', inplace=True)
```

## Call `record_linkage()`:
There is more than one way to achieve the same matching result.  But some ways are faster than others, depending on the data.

### Plot comparing Runtimes of `record_linkage()` calls with and without grouping on a test field having a varying number of unique values in a 10 000-row DataFrame
<center><img width="100%" src="https://raw.githubusercontent.com/ParticularMiner/red_string_grouper/master/Fuzzy_vs_Exact.png"></center>

### 1. Grouping by fields that are to be matched exactly
Note that those fields that have very few unique values distributed among a 
large number of rows, such as 'hasVideo' (2 unique values) and 'addressState' 
(51 unique values), can be specified as "fields that are to be matched exactly" 
(that is, in parameter `fields_2b_matched_exactly`) which can lead to a significant 
performance boost.

Behind the scenes, this allows `record_linkage()` to avoid using cosine-
similarity matching on these fields (which is time-consuming, since many 
matches are likely to be found), and instead group by these fields.

In this way, cosine-similarity matching can be performed only on the other 
fields (in parameter `fields_2b_matched_fuzzily`) for each group.

On the other hand, grouping by 'addressZipcode' (which has 6446 unique values) 
degrades performance, since the groups by this field are so many.  

To illustrate, the following call took 
&approx; 5 minutes to run:

```python
matches = record_linkage(
	df,
	fields_2b_matched_fuzzily=[('statusText', 0.8, 3, 1),
	                           ('address', 0.8, 3, 1)],
	fields_2b_matched_exactly=[('addressZipcode', 2),
	                           ('addressState', 4),
	                           ('hasVideo', 1)],
	hierarchical=True,
	max_n_matches=10000,
	similarity_dtype=np.float32,
	force_symmetries=False
)
```
whereas, the following call (which produces the same result) took &approx;8 seconds to run:


```python
matches = record_linkage(
	df,
	fields_2b_matched_fuzzily=[('statusText', 0.8, 3, 1),
	                           ('address', 0.8, 3, 1),
	                           ('addressZipcode', 0.999999, 3, 2)],
	fields_2b_matched_exactly=[('addressState', 4),
	                           ('hasVideo', 1)],
	hierarchical=True,
	max_n_matches=10000,
	similarity_dtype=np.float32,
	force_symmetries=False
)
```

Let's display the results:
```python
matches
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Exactly Matched Fields</th>
      <th colspan="9" halign="left">Fuzzily Matched Fields</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>addressState</th>
      <th>hasVideo</th>
      <th colspan="3" halign="left">statusText</th>
      <th colspan="3" halign="left">address</th>
      <th colspan="3" halign="left">addressZipcode</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Weighted Mean Similarity Score</th>
      <th></th>
      <th></th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
    </tr>
    <tr>
      <th>left_zpid</th>
      <th>right_zpid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2077667803</th>
      <th>2077679643</th>
      <td>1.000000</td>
      <td>WY</td>
      <td>false</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>Jlda Minor Sub Division LOT C, Buffalo, WY 82834</td>
      <td>1.000000</td>
      <td>Jlda Minor Subdivision LOT C, Buffalo, WY 82834</td>
      <td>82834</td>
      <td>1.0</td>
      <td>82834</td>
    </tr>
    <tr>
      <th>2075244057</th>
      <th>2075358943</th>
      <td>0.997100</td>
      <td>OH</td>
      <td>false</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>0 Township Road 118, Kimbolton, OH 43749</td>
      <td>0.973904</td>
      <td>Township Road 118, Kimbolton, OH 43749</td>
      <td>43749</td>
      <td>1.0</td>
      <td>43749</td>
    </tr>
    <tr>
      <th>2077676622</th>
      <th>2077676809</th>
      <td>0.993867</td>
      <td>ND</td>
      <td>false</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>4 55th St SE, Christine, ND 58015</td>
      <td>0.944802</td>
      <td>2 55th St SE, Christine, ND 58015</td>
      <td>58015</td>
      <td>1.0</td>
      <td>58015</td>
    </tr>
    <tr>
      <th>2077093064</th>
      <th>2078843498</th>
      <td>0.993328</td>
      <td>SD</td>
      <td>false</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>17 Sidney Park Rd, Custer, SD 57730</td>
      <td>0.939948</td>
      <td>Sidney Park Rd, Custer, SD 57730</td>
      <td>57730</td>
      <td>1.0</td>
      <td>57730</td>
    </tr>
    <tr>
      <th>150690392</th>
      <th>2076123604</th>
      <td>0.992909</td>
      <td>NJ</td>
      <td>false</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>5 Windsor Ln, Gladstone, NJ 07934</td>
      <td>0.936180</td>
      <td>0 Windsor Ln, Gladstone, NJ 07934</td>
      <td>7934</td>
      <td>1.0</td>
      <td>7934</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2070837516</th>
      <th>2072047318</th>
      <td>0.978032</td>
      <td>HI</td>
      <td>false</td>
      <td>New construction</td>
      <td>1.0</td>
      <td>New construction</td>
      <td>D12C Plan, Kaikoi at Hoopili</td>
      <td>0.802290</td>
      <td>D12B Plan, Kaikoi at Hoopili</td>
      <td>96706</td>
      <td>1.0</td>
      <td>96706</td>
    </tr>
    <tr>
      <th>305578084</th>
      <th>90035758</th>
      <td>0.977991</td>
      <td>MO</td>
      <td>false</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>210 N 17th St UNIT 203, Saint Louis, MO 63103</td>
      <td>0.801920</td>
      <td>210 N 17th St UNIT 1202, Saint Louis, MO 63103</td>
      <td>63103</td>
      <td>1.0</td>
      <td>63103</td>
    </tr>
    <tr>
      <th>2071195670</th>
      <th>88086529</th>
      <td>0.977983</td>
      <td>MI</td>
      <td>false</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>6533 E Jefferson Ave APT 426, Detroit, MI 48207</td>
      <td>0.801844</td>
      <td>6533 E Jefferson Ave APT 102E, Detroit, MI 48207</td>
      <td>48207</td>
      <td>1.0</td>
      <td>48207</td>
    </tr>
    <tr>
      <th>247263033</th>
      <th>247263136</th>
      <td>0.977941</td>
      <td>IA</td>
      <td>false</td>
      <td>New construction</td>
      <td>1.0</td>
      <td>New construction</td>
      <td>1 University Way #511, Iowa City, IA 52246</td>
      <td>0.801474</td>
      <td>1 University Way #503, Iowa City, IA 52246</td>
      <td>52246</td>
      <td>1.0</td>
      <td>52246</td>
    </tr>
    <tr>
      <th>2083656138</th>
      <th>2083656146</th>
      <td>0.977873</td>
      <td>IN</td>
      <td>false</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>3789 S Anderson Dr, Terre Haute, IN 47803</td>
      <td>0.800855</td>
      <td>3776 S Anderson Dr, Terre Haute, IN 47803</td>
      <td>47803</td>
      <td>1.0</td>
      <td>47803</td>
    </tr>
  </tbody>
</table>
<p>94 rows &times; 12 columns</p>
</div>



### 2. No grouping

The results above can be obtained in yet another way.  However, as mentioned 
above, it can take much longer to compute in cases where some fuzzily matched 
fields have very few uniques values.  

The following call took &approx;3 minutes 30 seconds to run:

```python
record_linkage(
	df,
	fields_2b_matched_fuzzily=[('statusText', 0.8, 3, 1),
	                           ('address', 0.8, 3, 1),
	                           ('addressZipcode', 0.999999, 3, 2),
	                           ('hasVideo', 0.999999, 3, 1),
	                           ('addressState', 0.999999, 2, 4)],
	hierarchical=True,
	max_n_matches=10000,
	similarity_dtype=np.float32,
	force_symmetries=False
)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">statusText</th>
      <th colspan="3" halign="left">address</th>
      <th colspan="3" halign="left">addressZipcode</th>
      <th colspan="3" halign="left">hasVideo</th>
      <th colspan="3" halign="left">addressState</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Weighted Mean Similarity Score</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
      <th>left</th>
      <th>similarity</th>
      <th>right</th>
    </tr>
    <tr>
      <th>left_zpid</th>
      <th>right_zpid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2077667803</th>
      <th>2077679643</th>
      <td>1.000000</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>Jlda Minor Sub Division LOT C, Buffalo, WY 82834</td>
      <td>1.000000</td>
      <td>Jlda Minor Subdivision LOT C, Buffalo, WY 82834</td>
      <td>82834</td>
      <td>1.0</td>
      <td>82834</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>WY</td>
      <td>1.0</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>2075244057</th>
      <th>2075358943</th>
      <td>0.997100</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>0 Township Road 118, Kimbolton, OH 43749</td>
      <td>0.973904</td>
      <td>Township Road 118, Kimbolton, OH 43749</td>
      <td>43749</td>
      <td>1.0</td>
      <td>43749</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>OH</td>
      <td>1.0</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>2077676622</th>
      <th>2077676809</th>
      <td>0.993867</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>4 55th St SE, Christine, ND 58015</td>
      <td>0.944802</td>
      <td>2 55th St SE, Christine, ND 58015</td>
      <td>58015</td>
      <td>1.0</td>
      <td>58015</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>ND</td>
      <td>1.0</td>
      <td>ND</td>
    </tr>
    <tr>
      <th>2077093064</th>
      <th>2078843498</th>
      <td>0.993328</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>17 Sidney Park Rd, Custer, SD 57730</td>
      <td>0.939948</td>
      <td>Sidney Park Rd, Custer, SD 57730</td>
      <td>57730</td>
      <td>1.0</td>
      <td>57730</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>SD</td>
      <td>1.0</td>
      <td>SD</td>
    </tr>
    <tr>
      <th>150690392</th>
      <th>2076123604</th>
      <td>0.992909</td>
      <td>Lot / Land for sale</td>
      <td>1.0</td>
      <td>Lot / Land for sale</td>
      <td>5 Windsor Ln, Gladstone, NJ 07934</td>
      <td>0.936180</td>
      <td>0 Windsor Ln, Gladstone, NJ 07934</td>
      <td>7934</td>
      <td>1.0</td>
      <td>7934</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>NJ</td>
      <td>1.0</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2070837516</th>
      <th>2072047318</th>
      <td>0.978032</td>
      <td>New construction</td>
      <td>1.0</td>
      <td>New construction</td>
      <td>D12C Plan, Kaikoi at Hoopili</td>
      <td>0.802290</td>
      <td>D12B Plan, Kaikoi at Hoopili</td>
      <td>96706</td>
      <td>1.0</td>
      <td>96706</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>HI</td>
      <td>1.0</td>
      <td>HI</td>
    </tr>
    <tr>
      <th>305578084</th>
      <th>90035758</th>
      <td>0.977991</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>210 N 17th St UNIT 203, Saint Louis, MO 63103</td>
      <td>0.801920</td>
      <td>210 N 17th St UNIT 1202, Saint Louis, MO 63103</td>
      <td>63103</td>
      <td>1.0</td>
      <td>63103</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>MO</td>
      <td>1.0</td>
      <td>MO</td>
    </tr>
    <tr>
      <th>2071195670</th>
      <th>88086529</th>
      <td>0.977983</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>6533 E Jefferson Ave APT 426, Detroit, MI 48207</td>
      <td>0.801844</td>
      <td>6533 E Jefferson Ave APT 102E, Detroit, MI 48207</td>
      <td>48207</td>
      <td>1.0</td>
      <td>48207</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>MI</td>
      <td>1.0</td>
      <td>MI</td>
    </tr>
    <tr>
      <th>247263033</th>
      <th>247263136</th>
      <td>0.977941</td>
      <td>New construction</td>
      <td>1.0</td>
      <td>New construction</td>
      <td>1 University Way #511, Iowa City, IA 52246</td>
      <td>0.801474</td>
      <td>1 University Way #503, Iowa City, IA 52246</td>
      <td>52246</td>
      <td>1.0</td>
      <td>52246</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>IA</td>
      <td>1.0</td>
      <td>IA</td>
    </tr>
    <tr>
      <th>2083656138</th>
      <th>2083656146</th>
      <td>0.977873</td>
      <td>Condo for sale</td>
      <td>1.0</td>
      <td>Condo for sale</td>
      <td>3789 S Anderson Dr, Terre Haute, IN 47803</td>
      <td>0.800855</td>
      <td>3776 S Anderson Dr, Terre Haute, IN 47803</td>
      <td>47803</td>
      <td>1.0</td>
      <td>47803</td>
      <td>false</td>
      <td>1.0</td>
      <td>false</td>
      <td>IN</td>
      <td>1.0</td>
      <td>IN</td>
    </tr>
  </tbody>
</table>
<p>94 rows &times; 16 columns</p>
</div>


One may choose to remove the field-values and output single-level column-
headings by setting hierarchical to `False`:  


```python
record_linkage(
	df,
	fields_2b_matched_fuzzily=[('statusText', 0.8, 3, 1),
	                           ('address', 0.8, 3, 1),
	                           ('addressZipcode', 0.999999, 3, 2),
	                           ('hasVideo', 0.999999, 3, 1),
	                           ('addressState', 0.999999, 2, 4)],
	hierarchical=False,
	max_n_matches=10000,
	similarity_dtype=np.float32,
	force_symmetries=False
)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Weighted Mean Similarity Score</th>
      <th>statusText</th>
      <th>address</th>
      <th>addressZipcode</th>
      <th>hasVideo</th>
      <th>addressState</th>
    </tr>
    <tr>
      <th>left_zpid</th>
      <th>right_zpid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2077667803</th>
      <th>2077679643</th>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2075244057</th>
      <th>2075358943</th>
      <td>0.997100</td>
      <td>1.0</td>
      <td>0.973904</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2077676622</th>
      <th>2077676809</th>
      <td>0.993867</td>
      <td>1.0</td>
      <td>0.944802</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2077093064</th>
      <th>2078843498</th>
      <td>0.993328</td>
      <td>1.0</td>
      <td>0.939948</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>150690392</th>
      <th>2076123604</th>
      <td>0.992909</td>
      <td>1.0</td>
      <td>0.936180</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2070837516</th>
      <th>2072047318</th>
      <td>0.978032</td>
      <td>1.0</td>
      <td>0.802290</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>305578084</th>
      <th>90035758</th>
      <td>0.977991</td>
      <td>1.0</td>
      <td>0.801920</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2071195670</th>
      <th>88086529</th>
      <td>0.977983</td>
      <td>1.0</td>
      <td>0.801844</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>247263033</th>
      <th>247263136</th>
      <td>0.977941</td>
      <td>1.0</td>
      <td>0.801474</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2083656138</th>
      <th>2083656146</th>
      <td>0.977873</td>
      <td>1.0</td>
      <td>0.800855</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>94 rows &times; 6 columns</p>
</div>

# Performance

## Plots of Runtimes of `record_linkage()` vs the number of blocks (`#blocks`) into which the left matrix-operand of the dataset (380 000 strings from sec__edgar_company_info.csv) was split before performing the string comparison.  As shown in the legend, each plot corresponds to the number of blocks into which the left matrix-operand was split.
<center><img width="100%" src="https://raw.githubusercontent.com/ParticularMiner/red_string_grouper/master/BlockSpaceExploration.png"></center>

String comparison, as implemented by `string_grouper`, is essentially matrix 
multiplication.  A DataFrame of strings is converted (tokenized) into a 
matrix.  Then that matrix is multiplied by itself.  

Here is an illustration of multiplication of two matrices ***M*** and ***D***:
![Block Matrix 1 1](https://user-images.githubusercontent.com/78448465/133109334-1a42cf7b-1780-42a9-a465-340464abe583.png)

It turns out that when the matrix (or DataFrame) is very large, the computer 
proceeds quite slowly with the multiplication (apparently due to the RAM being 
too full).  Some computers give up with an `OverflowError`.

To circumvent this issue, `red_string_grouper` allows to divide the DataFrame 
into smaller chunks (or blocks) and multiply the chunks one pair at a time 
instead to get the same result:

![Block Matrix 2 2](https://user-images.githubusercontent.com/78448465/133109377-76be7c85-a16d-4fcc-ade4-c2b475df6d0c.png)

But surprise ... the run-time of the process is sometimes drastically reduced 
as a result.  For example, the speed-up of the following call is about 200% 
(here, the DataFrame is divided into 4 blocks, that is, 
4 blocks on the left &times; 4 on the right) compared to the same call with
`n_blocks=(1, 1)` (the default) which is equivalent to `string_grouper`'s
`match_strings()`:

```python
companies = pd.read_csv('data/sec__edgar_company_info.csv')

# the following call produces the same result as 
# string_grouper using 
# match_strings(companies['Company Name'])
record_linkage(
	companies,
	fields_2b_matched_fuzzily=[('Company Name', 0.8, 3, 1)],
	fields_2b_matched_exactly=None,
	hierarchical=True,
	max_n_matches=10000,
	similarity_dtype=np.float32,
	force_symmetries=True,
	n_blocks=(4, 4)
)
```


Further exploration of the block number space shows that if the left operand 
is not split but the right operand is, then even more gains in speed can be 
made:

![Block Matrix 1 2](https://user-images.githubusercontent.com/78448465/133109548-672c22ed-297a-4bad-ab99-0957c0527163.png)

Here are some plots of the results of some experiments performed:

From the plot above, it can be seen that the optimum split-configuration 
(run-time &approx;3 minutes) is when the left operand is not split 
(#blocks = 1) and the right operand is split into six blocks (#nblocks = 6).

So what are the optimum block number values for a given DataFrame? That is 
anyone's guess, and the answer may vary from computer to computer.  

We however encourage the user to make judicious use of the `n_blocks` 
parameter.

