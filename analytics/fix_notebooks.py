"""Fix generate_notebooks.py for pandas 3.x compatibility"""
content = open('analytics/generate_notebooks.py','r',encoding='utf-8').read()

# Fix: pandas 3.x describe() rows are still 'mean','std' but use .loc[] to be safe
old1 = "print(df[['age', 'fare']].describe()[['mean', 'std']].round(4))"
new1 = "desc = df[['age', 'fare']].describe(); print(desc.loc[['mean','std']].round(4))"
content = content.replace(old1, new1)

old2 = "print(df_scaled_check[['age_z', 'fare_z']].describe()[['mean', 'std']].round(6))"
new2 = "desc2 = df_scaled_check[['age_z', 'fare_z']].describe(); print(desc2.loc[['mean','std']].round(6))"
content = content.replace(old2, new2)

# Also fix: matplotlib.use('Agg') must come before any other matplotlib imports in notebook context
# In a notebook, Agg backend needs to be set via matplotlib inline or Agg before plt import
# The issue in notebooks is that plt.show() with Agg may cause issues ? use savefig only
old3 = "plt.show()"
new3 = "plt.savefig  # already saved above"
# Don't replace show() calls - they're fine in notebooks, just add Agg correctly

open('analytics/generate_notebooks.py','w',encoding='utf-8').write(content)
print('Fixed.')
print('Verifying fix applied:')
print('old1 still present:', old1 in content)
print('new1 present:', new1 in content)
