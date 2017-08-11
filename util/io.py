import json
import cPickle
import os

#io functions of SCRC
def load_str_list(filename):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-1] for s in str_list]
    return str_list

def save_str_list(str_list, filename, end = '\n', mode = 'w'):
    str_list = [s+end for s in str_list]
    with open(filename, mode) as f:
        f.writelines(str_list)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        json.dump(json_obj, f, separators=(',\n', ':\n'))

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

def save_data(data, filename):
    with open(filename, 'wb') as f:
        cPickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data


#io functions of visual-concepts (attribute detector)
def save_variables(pickle_file_name, var, info = None, overwrite = False):
    if os.path.exists(pickle_file_name) and overwrite == False:
        raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
    if info == None:
        assert(type(var) == dict)
        with open(pickle_file_name, 'wb') as f:
            cPickle.dump(var, f, cPickle.HIGHEST_PROTOCOL)
    else:
        # Construct the dictionary
        assert(type(var) == list and type(info) == list);
        d = {}
        for i in xrange(len(var)):
            d[info[i]] = var[i]
        with open(pickle_file_name, 'wb') as f:
            cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            d = cPickle.load(f)
        return d
    else:
        raise Exception('{:s} dose not exist.'.format(pickle_file_name))
