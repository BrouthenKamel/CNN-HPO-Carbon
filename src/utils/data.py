import os
import json
import pandas as pd

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def surrogate_record(record):
    surr_record = {}
    surr_record['hp'] = record['hp']
    surr_record['train_accuracy'] = record['history']['epochs'][-1]['train_accuracy']
    surr_record['test_accuracy'] = record['history']['epochs'][-1]['test_accuracy']
    return surr_record

def assemble_surrogate_record(record_dir):
    records = []
    for file in os.listdir(record_dir):
        if file.endswith('.json'):
            print("Processing file: ", file)
            file_path = os.path.join(record_dir, file)
            data = load_json(file_path)
            surr_data = []
            for record in data:
                    surr_data.append(surrogate_record(record))
            records.extend(surr_data)
    save_json(records, os.path.join(record_dir, 'surrogate.json'))
    return records

def fix_indent(dir):
    for file in os.listdir(dir):
        if file.endswith('.json'):
            file_path = os.path.join(dir, file)
            data = load_json(file_path)
            save_json(data, file_path)
            
def flatten_record(record):
    hp = record['hp']
    flat_record = {}
    flat_record['initial_conv_hp_channels'] = hp['initial_conv_hp']['channels']
    flat_record['initial_conv_hp_kernel_size'] = hp['initial_conv_hp']['kernel_size']
    flat_record['initial_conv_hp_stride'] = hp['initial_conv_hp']['stride']
    flat_record['initial_conv_hp_activation'] = hp['initial_conv_hp']['activation']
    
    for i, block in enumerate(hp['inverted_residual_hps']):
        flat_record[f'block_{i}_expanded_channels'] = block['expanded_channels']
        flat_record[f'block_{i}_use_se'] = block['use_se']
        if block['use_se']:
            flat_record[f'block_{i}_se_squeeze_factor'] = block['se_hp']["squeeze_factor"]
            flat_record[f'block_{i}_se_activation'] = block['se_hp']["activation"]
        else:
            flat_record[f'block_{i}_se_squeeze_factor'] = 0
            flat_record[f'block_{i}_se_activation'] = 'NONE'
        flat_record[f'block_{i}_channels'] = block['conv_bn_activation_hp']['channels']
        flat_record[f'block_{i}_kernel_size'] = block['conv_bn_activation_hp']['kernel_size']
        flat_record[f'block_{i}_stride'] = block['conv_bn_activation_hp']['stride']
        flat_record[f'block_{i}_activation'] = block['conv_bn_activation_hp']['activation']
        
    flat_record['last_conv_upsample'] = hp['last_conv_upsample']
    flat_record['last_conv_hp_channels'] = hp['last_conv_hp']['channels']
    flat_record['last_conv_hp_kernel_size'] = hp['last_conv_hp']['kernel_size']
    flat_record['last_conv_hp_stride'] = hp['last_conv_hp']['stride']
    flat_record['last_conv_hp_activation'] = hp['last_conv_hp']['activation']
    
    flat_record["classifier_hp_neurons"] = hp['classifier_hp']['neurons']
    flat_record["classifier_hp_activation"] = hp['classifier_hp']['activation']
    flat_record["classifier_hp_dropout_rate"] = hp['classifier_hp']['dropout_rate']
    
    flat_record["train_accuracy"] = record['train_accuracy']
    flat_record["test_accuracy"] = record['test_accuracy']
    
    return flat_record
            
def create_surrogate_dataset(json_path):
    json_data = load_json(json_path)
    data = []
    for record in json_data:
        data.append(flatten_record(record))
        
    dataframe = pd.DataFrame(data)
    csv_path = json_path.replace('.json', '.csv')
    dataframe.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # assemble_surrogate_record('./dataset/')
    
    create_surrogate_dataset('./dataset/surrogate.json')