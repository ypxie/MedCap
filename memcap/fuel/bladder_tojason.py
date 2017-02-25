import sys
import json
import os
import shutil

sys.path.insert(0, '..')
from utils.local_utils import getfilelist

newfolder = '/home/yuanpuxie/DataSet/Bladder_Caption/Augmented/Annotation_new'
oldfolder = '/home/yuanpuxie/DataSet/Bladder_Caption/Augmented/Annotation'
new_filelist, new_filenames = getfilelist(newfolder,['.json'] )
old_filelist, old_filenames = getfilelist(oldfolder,['.json'] )

for new_idx, (new_fp, new_fn) in enumerate(zip(new_filelist, new_filenames)):
    with open(new_fp) as new_data_file:    
        new_data = json.load(new_data_file)    
        
        for old_idx, (old_fp, old_fn) in enumerate(zip(old_filelist, old_filenames)):
            
            if old_fn.startswith(new_fn):
                with open(old_fp, 'w') as f:
                     json.dump(new_data, f)
            