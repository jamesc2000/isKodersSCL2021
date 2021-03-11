import pandas as pd

#  data = pd.read_json('contacts.json')
data = pd.read_csv('contacts_test.csv')

order_id_link_buffer = []

current_id = 0
current_id_contacts = 0

#  current_id_email
#  current_id_phone
#  current_id_orderid

for Id in 

#  output =

duplicate_OrderId = data.duplicated(['OrderId'], keep=False)
duplicate_Phone = data.duplicated(['Phone'], keep=False)
duplicate_Email = data.duplicated(['Email'], keep=False)

print(duplicate_OrderId.head(5))
print(duplicate_Phone.head(5))
print(duplicate_Email.head(5))
print(data.shape)

print(data['Phone'][5])
