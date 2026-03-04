import os


def split_file(path, n_chunks):
    file_size = os.path.getsize(path)
    chunk_size = file_size // n_chunks

    with open(path, "rb") as f:
        for i in range(n_chunks):
            # Berechne die Größe für den letzten Chunk (Rest)
            current_chunk_size = chunk_size if i < n_chunks - 1 else (file_size - f.tell())

            chunk_data = f.read(current_chunk_size)

            # Speicherlogik
            out_path = f"{path}_part_{i}"
            with open(out_path, "wb") as out:
                out.write(chunk_data)


# Beispielaufruf
split_file(r"C:\Users\bestb\PycharmProjects\BestBrain\ESTRO 2025 - Abstract Book.pdf", 100)