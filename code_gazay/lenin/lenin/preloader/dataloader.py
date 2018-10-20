import torch
from collections import defaultdict


def dataloader(net, dataset, fields_name, records, batch_size, preload_opts, augmentors={}):
    active_fields = get_fields(net, dataset, fields_name)
    preprocessors = getattr(dataset, 'preprocessors', {})
    postprocessors = getattr(dataset, 'postprocessors', {})

    augmentors = filter_active_fields(augmentors, active_fields)
    preprocessors = filter_active_fields(preprocessors, active_fields)
    postprocessors = filter_active_fields(postprocessors, active_fields)

    preprocessors = separate_fields_and_concat_values(preprocessors)
    postprocessors = separate_fields_and_concat_values(postprocessors)

    def collate_fn(records):
        batch = { field: [] for field in active_fields }

        for record in records:
            transforms = { fields: augmentor.transformations() for fields, augmentor in augmentors.items() }
            transforms = separate_fields_and_concat_values(transforms)

            for field in set(active_fields):
                field_transforms = preprocessors[field] + transforms[field]
                accessor = getattr(dataset, field)
                if len(field_transforms):
                    value = accessor(record, compose(field_transforms))
                else:
                    value = accessor(record)
                value = compose(postprocessors[field])(value)
                batch[field].append(value)

        for field, values in batch.items():
            batch[field] = torch.utils.data.dataloader.default_collate(values)

        batch['records'] = records
        return batch

    return torch.utils.data.DataLoader(records, batch_size, collate_fn=collate_fn, **preload_opts)


def get_fields(net, dataset, fields_name):
    fields = getattr(net, fields_name, None) or getattr(net, 'fields', None) or dataset.fields
    fields = list(fields)
    if fields_name == 'train_fields' or fields_name == 'valid_fields':
        fields.append(dataset.target)
    return fields


def filter_active_fields(fields_dict, active_fields):
    result = {}
    for fields, values in fields_dict.items():
        fields = intersection(fields, active_fields)
        if len(fields):
            result[fields] = values
    return result


def separate_fields_and_concat_values(fields_dict):
    result = defaultdict(list)
    for fields, values in fields_dict.items():
        for field in fields:
            result[field] += to_list(values)
    return result


def intersection(list1, list2):
    return tuple(x1 for x1 in to_list(list1) if x1 in list2)


def to_list(arg):
    return isinstance(arg, (tuple, list)) and list(arg) or [arg]


def compose(functions):
    def inner(arg):
        for f in functions:
            if getattr(f, 'apply', None):
                arg = f.apply(arg)
            else:
                arg = f(arg)
        return arg
    return inner
