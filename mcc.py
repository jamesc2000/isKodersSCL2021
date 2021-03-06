import pandas as pd

data = pd.read_json('contacts.json')

order_id_link_buffer = []

current_id = 0
current_id_contacts = 0

#  current_id_email
#  current_id_phone
#  current_id_orderid

# output = 

duplicate_OrderId = data[data.duplicated(['OrderId'])]
duplicate_Phone = data[data.duplicated(['Phone'])]
duplicate_Email = data[data.duplicated(['Email'])]

print(data.describe())

print(data.head())

print(data.shape)

print(duplicate_OrderId.head(5))
print(duplicate_Phone.head(5))
print(duplicate_Email.head(5))

