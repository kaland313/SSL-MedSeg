import os
import glob
import shutil
from tqdm import tqdm
import pydoc


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs) 

    return pydoc.locate(object_type)(**kwargs)
    

def merge_pdfs(base_path=".", individual_pdfs_folder="", pdf_path_list=None, out_file_name = "merged.pdf", remove_files=False):
    from PyPDF2 import PdfFileMerger

    if pdf_path_list is None:
        inputs_folder = os.path.join(base_path, individual_pdfs_folder)
        pdf_paths = glob.glob(os.path.join(inputs_folder, "**", "*.pdf"), recursive=True)
        if not pdf_paths: 
            # No pdfs to merge were found
            return
    else:
        inputs_folder = os.path.commonpath(pdf_path_list)
        pdf_paths = pdf_path_list
    
    merger = PdfFileMerger()

    for pdf in tqdm(pdf_paths):
        merger.append(pdf)
        if remove_files:
            os.remove(pdf)

    merged_pdf_path = os.path.join(base_path, out_file_name)
    merger.write(merged_pdf_path)
    merger.close()
    print("Merged pdf saved to: ", os.path.abspath(merged_pdf_path))

    if remove_files:
        print("Removing folder", inputs_folder)
        shutil.rmtree(inputs_folder)
