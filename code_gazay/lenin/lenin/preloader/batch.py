class Batch:
    def __init__(self, records, fields):
        self.records = records

        for field, value in fields.items():
            setattr(self, field, value)

    def __len__(self):
        return len(self.records)
