import markdown
import json

def markdown_file_to_json(file_path):
    # Lire le contenu du fichier Markdown
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Convertir le texte Markdown en HTML
    html = markdown.markdown(markdown_text)

    # Extraire les lignes du tableau HTML
    table_rows = html.split('<tr>')[1:]
    table_rows = [row.split('</td>')[:-1] for row in table_rows]

    # Supprimer les balises HTML et les espaces vides
    table_data = [[cell.strip(' <td>') for cell in row] for row in table_rows]

    # Créer une liste de dictionnaires avec les données du tableau
    table_json = []
    for row in table_data:
        row_dict = {}
        for i, cell in enumerate(row):
            row_dict[f"col{i + 1}"] = cell
        table_json.append(row_dict)

    return table_json

# Chemin vers le fichier Markdown
markdown_file_path = 'tableau.md'

# Convertir le tableau Markdown en JSON
json_data = markdown_file_to_json(markdown_file_path)

# Écrire le JSON dans un fichier
with open('tableau.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print("Conversion terminée. Le fichier JSON a été créé.")