# Configure Google Drive API with OAuth

1. Login to your [Google Drive account](https://drive.google.com/)
2. Visit [ffhq-dataset](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP)
2. Visit the [Google Cloud Developer Console](https://console.cloud.google.com/cloud-resource-manager)
2. Click `Create Project` to create a new project
3. Visit the [Google Drive API](https://console.cloud.google.com/marketplace/product/google/drive.googleapis.com) and 
4. Click `Enable` to enable the Google Drive API
4. Select `OAuth consent screen` from the menu on the left hand side
5. Click on `Configure Consent Screen`
6. Click on `Create` and fill out the `App information` and `Developer contact information`
7. Select `Save and continue` until you reach the `Summary` page 
8. After the OAuth consent is done, select `Credentials` from the menu on the left hand side 
9. Select `OAuth consent screen` from the menu on the left hand side 
10. Click on `Create Credentials` then click on `OAuth client ID`
11. Select `Desktop app` and then click on `Create`
12. Once the OAuth client has been created, click on `Download JSON` to download the client configuration 
13. Rename the client configuration as `client_secrets.json`
14. Move `client_secrets.json` to the `a_view_from_somewhere` folder