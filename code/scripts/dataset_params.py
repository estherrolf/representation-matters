import pandas

dataset_params = {}

# goodreads
groups = ['history', 'fantasy']

pop_params = {}
# taken from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
pop_params['fantasy'] = {'n_reviews': 3424641,
                          'n_books': 258585}
pop_params['history'] = {'n_reviews': 2066193,
                          'n_books': 302935}
pop_param = 'n_reviews'

gamma = pop_params[groups[0]][pop_param] / (pop_params[groups[0]][pop_param]  + pop_params[groups[1]][pop_param] )

dataset_params_goodreads = {'gamma':gamma}
                           
dataset_params['goodreads'] = dataset_params_goodreads

# isic
isic_df = pandas.read_csv('../../data/isic/df_no_sonic_age_over_50_id_5_fold_splits.csv')
isic_group_key = 'age_over_50_id'
gamma = 1 - (isic_df[isic_df['fold'] == 'test'][isic_group_key]).mean()

dataset_params_isic = {'gamma':gamma}
                           
dataset_params['isic'] = dataset_params_isic

# adult
dataset_params_adult = {'gamma':0.5}
dataset_params['adult'] = dataset_params_adult

# mooc
# dataset_params_mooc = {'gamma':0.5} # this is for gender

mooc_df = pandas.read_csv('../../data/mooc/df_mooc_labels_5_post_secondary_fold_splits.csv')
mooc_group_key = 'post_secondary'
gamma = 1 - mooc_df[mooc_group_key].mean()
dataset_params_mooc = {'gamma': gamma} 
dataset_params['mooc'] = dataset_params_mooc


# cifar
cifar_params = {'gamma': 0.9}
dataset_params['cifar4'] = cifar_params 