import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def readability_score(filepath):
    def flatten_values(data):
        vals = []
        for key in data:
            if isinstance(data[key], dict):
                x = flatten_values(data[key])
                vals.extend(x)
            else:
                if isinstance(data[key], list):
                    vals.extend(data[key])
                else:
                    vals.append(data[key])
        return vals
    def count_words(text):
        words = text.split(None)
        return len(words)
    def count_sentences(text):
        sentences = 0
        end_sentence = ['.', '!', '?', ':', ';']
        for mark in end_sentence:
            sentences += text.count(mark)
        return sentences
    def count_syllables(text):
        syllables = 0
        vowels = ['a', 'e', 'i', 'o', 'u', 'y']
        rules = ['es', 'ed']
        words = text.split(None)
        for word in words:
            word.lower()
            if len(word) <= 3:
                syllables += 1
            else:
                end = word[-2:]
                if end in rules:
                    word = word[:-2]
                elif end == 'le':
                    word = word
                elif end[1] == 'e':
                    word = word[:-1]
                for vowel in vowels:
                    syllables += word.count(vowel)
        return syllables
    def flesch(text):
        n = count_sentences(text)
        w = count_words(text)
        l = count_syllables(text)
        if n == 0:
            n = 1
        if w == 0:
            w = 1
        f = 206.835 - 1.015 * (w/n) - 84.6 * (l/w)
        return f
    count = 0
    total_flesch = 0
    with open(filepath, 'r') as f:
        data = json.load(f)
        vals = flatten_values(data)
        for val in vals:
            if isinstance(val, str):
                count += 1
                total_flesch += flesch(val)
    # No strings in the metadata
    if count == 0:
        return 'no strings'
    return total_flesch/count

#Img
def completeness_score_img(filepath):
    def get_fields(data, field_dict):
        for field in data:
            if isinstance(data[field], dict) == False:
                field_dict[field] = []
            else:
                field_dict[field] = get_fields(data[field], {})
        return field_dict
    def flatten_fields(fields, fields_list, prefix):
        p = prefix
        for field in fields:
            prefix = p
            if fields[field] == []:
                if prefix == '':
                    fields_list.append(field)
                else:
                    fields_list.append(prefix + ':' + field)
            else:
                if prefix == '':
                    prefix = field
                else:
                    prefix = prefix + ':' + field
                flatten_fields(fields[field], fields_list, prefix)
        return fields_list
    max_fields = 4
    with open(filepath, 'r') as f:
        data = json.load(f)
        cur_fields = get_fields(data, {})
        cur_flattened_fields = flatten_fields(cur_fields, [], '')
        cur_number_of_fields = len(cur_flattened_fields)
    percent = (cur_number_of_fields/max_fields) * 100
    return percent

#JSONXML
def completeness_score_jsonxml(filepath):
    def get_fields(data, field_dict):
        for field in data:
            if isinstance(data[field], dict) == False:
                field_dict[field] = []
            else:
                field_dict[field] = get_fields(data[field], {})
        return field_dict
    def flatten_fields(fields, fields_list, prefix):
        p = prefix
        for field in fields:
            prefix = p
            if fields[field] == []:
                if prefix == '':
                    fields_list.append(field)
                else:
                    fields_list.append(prefix + ':' + field)
            else:
                if prefix == '':
                    prefix = field
                else:
                    prefix = prefix + ':' + field
                flatten_fields(fields[field], fields_list, prefix)
        return fields_list
    max_fields = 258
    with open(filepath, 'r') as f:
        data = json.load(f)
        cur_fields = get_fields(data, {})
        cur_flattened_fields = flatten_fields(cur_fields, [], '')
        cur_number_of_fields = len(cur_flattened_fields)
    percent = (cur_number_of_fields/max_fields) * 100
    return percent

#Keyword
def completeness_score_keyword(filepath):
    def get_fields(data, field_dict):
        for field in data:
            if isinstance(data[field], dict) == False:
                field_dict[field] = []
            else:
                field_dict[field] = get_fields(data[field], {})
        return field_dict
    def flatten_fields(fields, fields_list, prefix):
        p = prefix
        for field in fields:
            prefix = p
            if fields[field] == []:
                if prefix == '':
                    fields_list.append(field)
                else:
                    fields_list.append(prefix + ':' + field)
            else:
                if prefix == '':
                    prefix = field
                else:
                    prefix = prefix + ':' + field
                flatten_fields(fields[field], fields_list, prefix)
        return fields_list
    max_fields = 2045
    with open(filepath, 'r') as f:
        data = json.load(f)
        cur_fields = get_fields(data, {})
        cur_flattened_fields = flatten_fields(cur_fields, [], '')
        cur_number_of_fields = len(cur_flattened_fields)
    percent = (cur_number_of_fields/max_fields) * 100
    return percent

#NetCDF
def completeness_score_netcdf(filepath):
    def get_fields(data, field_dict):
        for field in data:
            if isinstance(data[field], dict) == False:
                field_dict[field] = []
            else:
                field_dict[field] = get_fields(data[field], {})
        return field_dict
    def flatten_fields(fields, fields_list, prefix):
        p = prefix
        for field in fields:
            prefix = p
            if fields[field] == []:
                if prefix == '':
                    fields_list.append(field)
                else:
                    fields_list.append(prefix + ':' + field)
            else:
                if prefix == '':
                    prefix = field
                else:
                    prefix = prefix + ':' + field
                flatten_fields(fields[field], fields_list, prefix)
        return fields_list
    max_fields = 2849
    with open(filepath, 'r') as f:
        data = json.load(f)
        cur_fields = get_fields(data, {})
        cur_flattened_fields = flatten_fields(cur_fields, [], '')
        cur_number_of_fields = len(cur_flattened_fields)
    percent = (cur_number_of_fields/max_fields) * 100
    return percent

#Tabular
def completeness_score_tabular(filepath):
    def get_fields(data, field_dict):
        for field in data:
            if isinstance(data[field], dict) == False:
                field_dict[field] = []
            else:
                field_dict[field] = get_fields(data[field], {})
        return field_dict
    def flatten_fields(fields, fields_list, prefix):
        p = prefix
        for field in fields:
            prefix = p
            if fields[field] == []:
                if prefix == '':
                    fields_list.append(field)
                else:
                    fields_list.append(prefix + ':' + field)
            else:
                if prefix == '':
                    prefix = field
                else:
                    prefix = prefix + ':' + field
                flatten_fields(fields[field], fields_list, prefix)
        return fields_list
    max_fields = 7407
    with open(filepath, 'r') as f:
        data = json.load(f)
        cur_fields = get_fields(data, {})
        cur_flattened_fields = flatten_fields(cur_fields, [], '')
        cur_number_of_fields = len(cur_flattened_fields)
    percent = (cur_number_of_fields/max_fields) * 100
    return percent

def tfidf_score(filepath):
    def flatten_values(data):
        vals = []
        for key in data:
            if isinstance(data[key], dict):
                x = flatten_values(data[key])
                vals.extend(x)
            else:
                if isinstance(data[key], list):
                    vals.extend(data[key])
                else:
                    vals.append(data[key])
        return vals
    text = ''
    with open (filepath, 'r') as f:
        data = json.load(f)
        vals = flatten_values(data)
        for val in vals:
            if isinstance(val, str):
                print(val)
                text = text + ' ' + val
    
    vectorizer = TfidfVectorizer()
    '''
    #try:
    vectors = vectorizer.fit_transform([text])
    print(text)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns = feature_names)
    total = 0
    for val in denselist:
        total += val
    return total/len(feature_names)
    #except Exception as e:
        #print(e)
        #return 0
    '''
    try:
        vectors = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns = feature_names)
        total = 0
        for col in df.columns:
            vals = df.loc[:,col]
            for val in vals:
                total += val
        return total/len(df.columns)
    except ValueError:
        return 0

score = tfidf_score('/home/cc/CDIACMetadataExtract/CDIACTabularExtracted/PACIFICA1205.csvTabXtract50.json')
print(score)