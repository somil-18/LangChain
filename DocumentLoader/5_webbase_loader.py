# we can also pass list of URLs

from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/lg-8-5-kg-5-star-roller-jet-pulsator-soak-wind-dry-collar-scrubber-semi-automatic-top-load-washing-machine-maroon-white/p/itmb46d8b1694357?pid=WMNGMA88AJE8S3JF&lid=LSTWMNGMA88AJE8S3JFCBCELW&marketplace=FLIPKART&store=j9e%2Fabm%2F8qx&srno=b_1_3&otracker=nmenu_sub_TVs%20%26%20Appliances_0_Semi%20Automatic%20Top%20Load&fm=organic&iid=ad1e2c22-6a06-4ada-854d-de91bf4efe6d.WMNGMA88AJE8S3JF.SEARCH&ppt=browse&ppn=browse&ssid=usqj4mdmds0000001763559213799'

loader = WebBaseLoader(url)

docs = loader.load()

# Print number of documents extracted
print(len(docs))
print('\n')

# Print the content of the first document (raw HTML/text extracted)
print(docs[0])
print('\n')
