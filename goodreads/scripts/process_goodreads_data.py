from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import sys
import scipy.sparse

sys.path.append('../../code/scripts')
from dataset_chunking_fxns import add_stratified_kfold_splits

data_dir = '../../data'
goodreads_data_dir = os.path.join(data_dir, 'goodreads')

def parse_reviews(input_fn):

    data_full = pd.read_json(input_fn,lines=True)#

    return data_full

def tfidf_features(reviews, max_features=5000, use_stopwords=False):
    # reviews is a list of reviews
    
    if use_stopwords:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    else:
        vectorizer = TfidfVectorizer(max_features=max_features)
    
    X = vectorizer.fit_transform(reviews)
    
    return X, vectorizer


def split_test(n_train, n_test, n_total, random_seed):
    rs = np.random.RandomState(random_seed)
    
    n = n_train + n_test 
    shuffled_idxs = rs.choice(n, n, replace=False)

    train_idxs = shuffled_idxs[:n_train]
    test_idxs = shuffled_idxs[n_train:n_train+n_test]

    return train_idxs, test_idxs

def add_test_splits(data, frac_test):
    n_total = len(data)
    n_test = int(frac_test*n_total)
    n_train = n_total - n_test
    
    train_idxs, test_idxs = split_test(n_train, n_test, n_total, 0)
    
    splits = np.array(['empty_string'] * n_total)
    splits[train_idxs] = 'train'
    splits[test_idxs] = 'test'

    # add splits
    data.insert(0,'fold',splits)
    return data


# just subset randomly
def subset_data_randomly(data_full, n_samples):
    rs = np.random.RandomState(0)
    smaller_idxs = rs.choice(len(data_full), int(n_samples), replace=False)
    
    return data_full.iloc[smaller_idxs]

def count_reviews_by_book(data_full):
    data_by_book_id = data_full.groupby('book_id').count()['rating']
    book_ids = data_by_book_id.index
    book_rating_cts = data_by_book_id.values
    
    return book_ids, book_rating_cts

def subset_data_top_k_books(data_full, k):
    # find number of reviews per book
    book_ids, book_rating_cts = count_reviews_by_book(data_full)
    
    # find book ids of the most-reviewed books
    book_ids_big = book_ids[np.argsort(book_rating_cts)[-k:]]
    
    # return dataframe corresponding to just these books
    locs_book_ids_big = np.where(data_full['book_id'].apply(lambda x: x in book_ids_big))[0]
    return data_full.iloc[locs_book_ids_big]

def aggregate_reviews(genres, 
                      data_by_genre, 
                      csv_name_pattern,
                      n_per_genre=None,
                      k=None, 
                      frac_test = 0.2,
                      n_kfold_splits=5):
   
    # 1. take out any books in genres
    book_ids_overlapping = np.intersect1d(data_by_genre[0]['book_id'], 
                                          data_by_genre[1]['book_id'])
                                          
    print('before removing nans: {0} overlapping book ids to remove'.format(len(book_ids_overlapping)))
                                     
    data_by_genre_deduped = []
    for i,data_this in enumerate(data_by_genre):
        print(genres[i])
        # remove nans or empty strings
        print('input dataset size: ', len(data_this))
        data_this.replace('', float('Nan'), inplace=True)
        data_this.replace('null', float('Nan'), inplace=True)
        # don't allow 0's in the rating column
        data_this['rating'] = data_this['rating'].replace(0,float('NaN'))
        
        data_this.dropna(subset=['book_id','review_text','rating'], inplace=True)
        print('after removing nans/invalid: ', len(data_this))
        
        # remove overlaps
        this_overlapping_locs = np.where(data_this['book_id'].apply(lambda x: x not in book_ids_overlapping))[0]
        data_this_dedup = data_this.iloc[this_overlapping_locs]

        data_by_genre_deduped.append(data_this_dedup)
        print('after deduplicating: ', len(data_this))
        
    data_to_consider = data_by_genre_deduped
    if not k is None:
        data_by_genre_top_k = []
        for data_this in data_by_genre_deduped:
            data_by_genre_top_k.append(subset_data_top_k_books(data_this, k))
    
        print('after subsetting to top {} most reviewed books per genre:'.format(k))    
        for i,data_this in enumerate(data_by_genre_top_k):
            print("{0} :".format(genres[i]),len(data_this))
            
        data_to_consider = data_by_genre_top_k
        
    # if no max size given, pick the size of the smallest
    if n_per_genre is None:
        n_per_genre = np.min([len(x) for x in data_to_consider])
    
    
    # subset and add genre and test splits
    data_by_genre_smaller = []
    #for i,data_this in enumerate(data_by_genre_deduped):
    for i,data_this in enumerate(data_to_consider):
        
        data_this_smaller = subset_data_randomly(data_this, n_per_genre)

        # add train/test splits
        data_this_smaller = add_test_splits(data_this_smaller, frac_test = frac_test)

        # add groups
        data_this_smaller.insert(0,'genre_name', genres[i])
        data_this_smaller.insert(0,'genre', i*np.ones(len(data_this_smaller)))
        data_by_genre_smaller.append(data_this_smaller)
        
        
    print('in final dataset:')
    for i,data_this in enumerate(data_by_genre_smaller):
        print('mean rating for {0}: {1:.3f}'.format(genres[i],
          data_this['rating'].mean()))

    print('total num reviews for {0}: '.format(genres[i]),
          len(data_this['rating']))
    
    # concatenate
    data_both = pd.concat(data_by_genre_smaller, ignore_index=True)
        
    fn_save = os.path.join(goodreads_data_dir, csv_name_pattern)
    # add x idxs to match feaures
    data_both['X_idxs'] = np.arange(len(data_both))
    data_both.to_csv(fn_save, index=False)  
       
    # create and save features
    
    features, vectorizer = tfidf_features(list(data_both['review_text']), 
                                        max_features=2000,
                                        use_stopwords=False)
    
    # save features
    features_fn_save = fn_save.replace('.csv', '_features_2k.npz')
    print('saving tfidf features in ', features_fn_save)
    scipy.sparse.save_npz(features_fn_save, features)
    # add stratified kfold splits and save
    data_both_with_cv_splits = add_stratified_kfold_splits(fn_save,
                                                           'genre',
                                                            num_splits=n_kfold_splits,
                                                            overwrite=False)
    
    
    
    return data_both_with_cv_splits

def main():
    # by default do history and fanstasy    
    reviews_fn_history = 'goodreads_reviews_history_biography.json.gz'
    reviews_fn_fantasy = 'goodreads_reviews_fantasy_paranormal.json.gz'

    fn_history = os.path.join(goodreads_data_dir,reviews_fn_history)
    fn_fantasy = os.path.join(goodreads_data_dir,reviews_fn_fantasy)
    
    print('parsing reviews')
    data_history = parse_reviews(fn_history)
    data_fantasy = parse_reviews(fn_fantasy)

    print('aggregating to one dataframe')
    genres = ['history', 'fantasy']
    data_by_genre = [data_history, data_fantasy]
    
    csv_filename = 'goodreads_{0}_{1}.csv'.format(genres[0], genres[1])
    
    # this function call will save the csv file 
    data_history_fantasy = aggregate_reviews(genres, 
                                            data_by_genre,
                                            csv_filename,
                                            k=100, 
                                            n_per_genre=62500,
                                            frac_test = 0.2,
                                            n_kfold_splits = 5)
    return data_history_fantasy

if __name__ == '__main__':
    main()
    
    
    
    
    
    