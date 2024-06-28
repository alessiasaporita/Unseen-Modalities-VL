import torch
import io
import pyarrow as pa
import os

from PIL import Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        preprocess: None,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=77,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,        
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        preprocess: preprocess transform of CLIP
        """
        super().__init__()

        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.preprocess = preprocess
        #['image', 'text', 'label', 'image_id', 'split']

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ] 

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "": #plots, text
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist() #all plots/texts of the table as a list of arrays (array of n element = plots/texts)
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict() #id : (id_text, index) index =[0, n plots for that texts]

        if 'text_aug' in self.table.column_names:
            self.aug_texts = self.table['text_aug'].to_pandas().tolist()
        else:
            self.aug_texts = None
        
        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)
        
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index] #index img, index caption
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [self.preprocess(image)]
        #image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]

        return {
            "text": text,
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }
    

    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()]) #keys, image, label, text
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key] #all images of the batch, [256 x [image = (3,384,384)]]
            view_size = len(img[0]) #1

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ] #[images = (256, 3, 384, 384)], array with one element, the tensor with the batch of images

            for bi in range(batch_size):
                orig_batch = img[bi] #[image = (3, 384, 384)]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi] #(3,384,384)
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d for d in dict_batch[txt_key]] for txt_key in txt_keys]
            for i, txt_key in enumerate(txt_keys):
                texts = (
                    [d for d in dict_batch[txt_key]],
                )
                dict_batch[txt_key] = texts[0]

        return dict_batch

