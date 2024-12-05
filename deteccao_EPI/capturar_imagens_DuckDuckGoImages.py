from bing_image_downloader import downloader

# Defina o termo de busca e a pasta de destino
query_string = "pessoa usando capacete com abafador de ruídos (EPI)"  # Palavra-chave de busca
destino = "E:/Desktop/Pós PUC/EPI-Detection-using-yolov8-main/dataset"  # Pasta onde as imagens serão salvas

print("Iniciando download...")

# Tente baixar as imagens
try:
    downloader.download(query_string, limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    print("Downloads concluídos!")
except Exception as e:
    print(f"Erro durante o download: {e}")




