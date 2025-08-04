loading annotations into memory...
Done (t=0.07s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!

=== Initial Dataset Inspection ===
Training dataset keys: ['info', 'images', 'licenses', 'annotations', 'categories']
Test dataset keys: ['info', 'images', 'licenses', 'annotations', 'categories']

Training set:
  Number of images: 8981
  Number of annotations: 23699
  Number of categories: 79

Test set:
  Number of images: 325

Sample image entry:
{
  "id": 1,
  "width": 720,
  "height": 368,
  "file_name": "67cda248-6d28-4801-9c4e-e4525189ea38.png",
  "license": 0,
  "flickr_url": "https://database.fathomnet.org/static/m3/framegrabs/Ventana/images/3317/01_17_26_19.png",
  "coco_url": "https://database.fathomnet.org/static/m3/framegrabs/Ventana/images/3317/01_17_26_19.png",
  "date_captured": "2008-12-19 19:46:56"
}

Sample annotation entry:
{
  "id": 1,
  "image_id": 1,
  "category_id": 71,
  "segmentation": [],
  "area": 943.0,
  "bbox": [
    491.0,
    254.0,
    23.0,
    41.0
  ],
  "iscrowd": 0
}

Sample category entry:
{
  "id": 1,
  "name": "Abyssocucumis abyssorum",
  "supercategory": ""
}

=== Category Analysis ===
Number of categories: 79

First 10 categories:
   id                     name
0   1  Abyssocucumis abyssorum
1   2            Acanthascinae
2   3            Acanthoptilum
3   4               Actinernus
4   5               Actiniaria
5   6           Actinopterygii
6   7                Amphipoda
7   8  Apostichopus leukothele
8   9             Asbestopluma
9  10   Asbestopluma monticola

Annotation counts per category (top 10):
                      name  count
0             Sebastolobus    300
1  Apostichopus leukothele    300
2              Scotoplanes    300
3               Keratoisis    300
4              Munnopsidae    300
5      Scotoplanes globosa    300
6     Chionoecetes tanneri    300
7              Hippasteria    300
8               Munidopsis    300
9               Serpulidae    300

=== Taxonomic Hierarchy Analysis ===
Retrieving taxonomic information...
Error retrieving data for Abyssocucumis abyssorum: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Acanthascinae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Acanthoptilum: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Actinernus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Actiniaria: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Actinopterygii: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Amphipoda: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Apostichopus leukothele: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Asbestopluma: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Asbestopluma monticola: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Asteroidea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Benthocodon pedunculata: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Brisingida: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Caridea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Ceriantharia: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Chionoecetes tanneri: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Chorilia longipes: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Corallimorphus pilatus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Crinoidea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Delectopecten: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Elpidia: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Farrea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Florometra serratissima: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Funiculina: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Gastropoda: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Gersemia juliepackardae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Heterocarpus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Heterochone calyx: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Heteropolypus ritteri: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Hexactinellida: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Hippasteria: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Holothuroidea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Hormathiidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Isidella tentaculum: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Isididae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Isosicyonis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Keratoisis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Liponema brevicorne: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Lithodidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Mediaster aequalis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Merluccius productus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Metridium farcimen: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Microstomus pacificus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Munidopsis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Munnopsidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Mycale: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Octopus rubescens: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Ophiacanthidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Ophiuroidea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Paelopatides confundens: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Pandalus amplus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Pandalus platyceros: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Pannychia moseleyi: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Paragorgia: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Paragorgia arborea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Paralomis multispina: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Parastenella: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Peniagone: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Pennatula phosphorea: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Porifera: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Psathyrometra fragilis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Psolus squamatus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Ptychogastria polaris: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Pyrosoma atlanticum: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Rathbunaster californicus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Scleractinia: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Scotoplanes: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Scotoplanes globosa: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Sebastes: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Sebastes diploproa: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Sebastolobus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Serpulidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Staurocalyptus: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Strongylocentrotus fragilis: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Terebellidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Tunicata: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Umbellula: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Vesicomyidae: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'
Error retrieving data for Zoantharia: module 'fathomnet.api.worms' has no attribute 'find_taxa_by_scientific_name'

Taxonomic rank distribution:
Empty DataFrame
Columns: [rank, count]
Index: []

=== Annotation Analysis ===

Annotations per image statistics:
  Mean: 2.64
  Median: 1.00
  Min: 1
  Max: 107

ROI area statistics:
  Mean: 39296.42
  Median: 4950.00
  Min: 63.00
  Max: 2075604.00

ROI aspect ratio statistics:
  Mean: 1.16
  Median: 1.08
  Min: 0.12
  Max: 8.83