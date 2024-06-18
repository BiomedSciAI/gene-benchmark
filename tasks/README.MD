# Task retrieval

Our package allows users to easily evaluate the gene encodings on hundreds of tasks from varying sources and types. The package does not include the task data itself to avoid redistributing data from various sources with various public licenses. However, we provide an easy [command line interface](../scripts/tasks_retrival/) to download and populate a folder with the tasks in the appropriate format. The list of tasks and their descriptions are available in an [excel file](task_descriptions.xlsx).

Note that the interface allows to download data from various sources. To do so the user needs to explicitly allow this. Use this option only if you trust the URLs appearing in the scripts.

Also, note that the gene-benchmark package includes a unit test that assumes that the tasks from the scripts are populated into this folder. This test is disabled by default and is mostly used within the GitHub actions to ensure the pipeline is intact.
