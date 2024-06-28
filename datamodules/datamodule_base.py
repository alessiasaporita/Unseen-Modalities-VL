import torch
from torch.utils.data import DataLoader


class BaseDataLoader():
    def __init__(self, preprocess_train, preprocess_val, preprocess_test, args):
        super().__init__()

        self.data_dir = args.data_root
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.preprocess_test = preprocess_test
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

        self.image_size = args.image_size
        self.max_text_len = args.max_text_len
        self.draw_false_image = args.draw_false_image
        self.draw_false_text = args.draw_false_text
        self.image_only = args.image_only
        
        # construct missing modality info
        self.missing_info = {
            'ratio' : {'train': args.missing_ratio, 'val': args.missing_ratio, 'test': args.missing_ratio}, 
            'type' : {'train': args.missing_type, 'val': args.missing_type, 'test': args.missing_type},
            'both_ratio' : args.both_ratio,
            'missing_table_root': args.missing_table_root,
            'simulate_missing' : args.simulate_missing
        }

        # for bash execution
        if args.test_ratio is not None:
            self.missing_info['ratio']['val'] = args.test_ratio
            self.missing_info['ratio']['test'] = args.test_ratio
        if args.test_type is not None:
            self.missing_info['type']['val'] = args.test_type
            self.missing_info['type']['test'] = args.test_type
        
        #self.train_transform_keys = ["pixelbert"] 
        #self.val_transform_keys = ["pixelbert"] 
        """
        self.train_transform_keys = (
            ["default_train"]
            if len(args.train_transform_keys) == 0
            else args.train_transform_keys
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(args.val_transform_keys) == 0
            else args.val_transform_keys
        )
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        """
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            #self.train_transform_keys,
            split="train",
            preprocess=self.preprocess_train,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            #self.val_transform_keys,
            split="val",
            preprocess=self.preprocess_val,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                #self.val_transform_keys,
                split="val",
                preprocess=self.preprocess_val,
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            #self.val_transform_keys,
            split="val",
            preprocess=self.preprocess_val,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            #self.val_transform_keys,
            split="test",
            preprocess=self.preprocess_test,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

    def setup(self):
        if not self.setup_flag:
            #create table with indexes of imgs marked as missing for train, test e val
            self.set_train_dataset() 
            self.set_val_dataset()
            self.set_test_dataset()

            #self.train_dataset.tokenizer = self.tokenizer
            #self.val_dataset.tokenizer = self.tokenizer
            #self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader


    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
