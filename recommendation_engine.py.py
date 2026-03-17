import pandas as pd
import numpy as np
from numpy.linalg import norm
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
from scipy.sparse import csr_matrix
import time
import re
import os

import warnings

# Gereksiz pandas uyarılarını kapatalım
warnings.filterwarnings('ignore')

# =============================================================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# =============================================================================
def load_data():
    print("📂 Veriler yükleniyor...")
    books = pd.read_csv('books.csv')
    ratings = pd.read_csv('ratings.csv')

    # Kitap detaylarını hızlı erişim için sözlüğe çevir
    books_info = books.set_index('book_id')[['title', 'authors', 'original_title']].to_dict('index')

    # KATALOG KAPSAMI İÇİN FİLTRE DENGELENDİ
    min_book_ratings = 200
    min_user_ratings = 10

    counts_b = ratings['book_id'].value_counts()
    ratings = ratings[ratings['book_id'].isin(counts_b[counts_b > min_book_ratings].index)]

    counts_u = ratings['user_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(counts_u[counts_u > min_user_ratings].index)]

    print(f"✅ Filtreleme Tamamlandı: {len(ratings)} etkileşim üzerinden işlem yapılacak.")
    return books, ratings, books_info

# =============================================================================
# 1b. TAG / GENRE VERİLERİNİ YÜKLEME VE PROFİL OLUŞTURMA
# =============================================================================
def load_tags(books_df):
    """Tag ve book_tags verilerini yükler, gürültülü tag'leri temizler."""
    print("🏷️  Tag verileri yükleniyor...")
    tags = pd.read_csv('tags.csv')
    book_tags = pd.read_csv('book_tags.csv')

    # goodreads_book_id -> book_id eşlemesi
    gr_to_bid = books_df.set_index('goodreads_book_id')['book_id'].to_dict()
    book_tags['book_id'] = book_tags['goodreads_book_id'].map(gr_to_bid)
    book_tags = book_tags.dropna(subset=['book_id'])
    book_tags['book_id'] = book_tags['book_id'].astype(int)

    # Tag isimlerini birleştir
    book_tags = book_tags.merge(tags, on='tag_id', how='left')
    book_tags = book_tags.dropna(subset=['tag_name'])

    # Gürültülü tag'leri filtrele (yıldız, okuma durumu, kişisel raflar)
    noise_patterns = [
        r'^\d+-star', r'^to-read', r'^currently-reading', r'^read-in-',
        r'^favorites$', r'^owned', r'^my-', r'^i-',
        r'^kindle', r'^audiobook', r'^ebook', r'^library',
        r'^book-club', r'^wish-list', r'^want-to-read',
        r'^dnf', r'^gave-up', r'^abandoned',
        r'^\d+$', r'^--', r'^\d+-'
    ]
    noise_regex = '|'.join(noise_patterns)
    book_tags = book_tags[~book_tags['tag_name'].str.match(noise_regex, case=False)]

    # Kitap başına minimum tag sayısı filtresi (çok düşük count'ları at)
    book_tags = book_tags[book_tags['count'] >= 5]

    print(f"✅ {book_tags['book_id'].nunique()} kitap için {book_tags['tag_id'].nunique()} benzersiz tag yüklendi.")
    return book_tags

def build_tag_profiles(book_tags_df):
    """TF-IDF ağırlıklı kitap-tag profil matrisi oluşturur."""
    print("🔧 Tag profil matrisi oluşturuluyor...")

    # Benzersiz kitap ve tag listesi
    unique_books = sorted(book_tags_df['book_id'].unique())
    unique_tags = sorted(book_tags_df['tag_id'].unique())

    book_to_idx = {bid: i for i, bid in enumerate(unique_books)}
    tag_to_idx = {tid: i for i, tid in enumerate(unique_tags)}

    # Sparse matris için veri hazırla
    rows, cols, values = [], [], []
    for _, row in book_tags_df.iterrows():
        bid = row['book_id']
        tid = row['tag_id']
        if bid in book_to_idx and tid in tag_to_idx:
            rows.append(book_to_idx[bid])
            cols.append(tag_to_idx[tid])
            values.append(row['count'])

    # Sparse count matrisi
    n_books = len(unique_books)
    n_tags = len(unique_tags)
    count_matrix = csr_matrix((values, (rows, cols)), shape=(n_books, n_tags))

    # TF-IDF dönüşümü: her tag'in IDF'si = log(N / df_tag)
    # df_tag = bu tag'i kullanan kitap sayısı
    df = np.array((count_matrix > 0).sum(axis=0)).flatten().astype(float)
    df[df == 0] = 1  # sıfıra bölme koruması
    idf = np.log(n_books / df)

    # Her satırı normalize et (L2 norm)
    from scipy.sparse import diags
    tfidf_matrix = count_matrix.multiply(idf)  # broadcast IDF
    row_norms = np.sqrt(np.array(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).flatten())
    row_norms[row_norms == 0] = 1
    inv_norms = diags(1.0 / row_norms)
    tfidf_matrix = inv_norms @ tfidf_matrix

    # Tag isimlerini indeksle
    tag_id_to_name = book_tags_df.drop_duplicates('tag_id').set_index('tag_id')['tag_name'].to_dict()
    idx_to_tag_name = {tag_to_idx[tid]: tag_id_to_name.get(tid, '?') for tid in unique_tags}

    print(f"✅ Tag profil matrisi hazır: {n_books} kitap × {n_tags} tag")
    return tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name

# =============================================================================
# 2. GERÇEKÇİ METRİK HESAPLAMA FONKSİYONU
# =============================================================================
def calculate_metrics(model, ratings_df, k=5):
    """Katalog Kapsamı ve Precision@K değerlerini gerçekçi hold-out yöntemiyle hesaplar."""
    print("📏 Metrikler hesaplanıyor (hold-out Precision@K)...")

    unique_users = ratings_df['user_id'].unique()
    np.random.seed(42)

    # --- Precision@K: 500 kullanıcı örneklemi ---
    prec_sample_size = min(500, len(unique_users))
    prec_sample_users = np.random.choice(unique_users, prec_sample_size, replace=False)

    # --- Coverage: 2000 kullanıcı örneklemi, top-20 öneri ---
    cov_sample_size = min(2000, len(unique_users))
    cov_sample_users = np.random.choice(unique_users, cov_sample_size, replace=False)
    coverage_k = 20  # Coverage için daha geniş top-k

    all_books = ratings_df['book_id'].unique()
    all_books_set = set(all_books)
    recommended_items = set()
    precision_scores = []

    trainset = model.trainset
    prec_sample_set = set(prec_sample_users)

    for user_id in cov_sample_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        read_ids = set(user_ratings['book_id'])

        # --- Precision@K (sadece precision örneklemindeki kullanıcılar için) ---
        if user_id in prec_sample_set:
            liked = set(user_ratings[user_ratings['rating'] >= 4]['book_id'])
            if len(liked) >= 2:
                liked_list = list(liked)
                np.random.shuffle(liked_list)
                n_holdout = max(1, len(liked_list) // 5)
                holdout_set = set(liked_list[:n_holdout])

                unread_ids = list(all_books_set - read_ids)
                neg_count = min(99, len(unread_ids))
                neg_samples = set(np.random.choice(unread_ids, neg_count, replace=False))
                precision_pool = list((neg_samples - holdout_set) | holdout_set)

                prec_preds = [(bid, model.predict(user_id, bid).est) for bid in precision_pool]
                prec_preds.sort(key=lambda x: x[1], reverse=True)
                top_k_prec = [p[0] for p in prec_preds[:k]]

                hits = len(set(top_k_prec) & holdout_set)
                precision_scores.append(hits / k)

        # --- Katalog Kapsamı: vektörel SVD tahmini (tüm kitaplar) ---
        try:
            inner_uid = trainset.to_inner_uid(user_id)
        except ValueError:
            continue

        scores = model.qi.dot(model.pu[inner_uid])

        for bid in read_ids:
            try:
                scores[trainset.to_inner_iid(bid)] = -999
            except ValueError:
                pass

        top_k_inner = np.argsort(scores)[::-1][:coverage_k]
        for iid in top_k_inner:
            recommended_items.add(trainset.to_raw_iid(iid))

    coverage = (len(recommended_items) / len(all_books)) * 100
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0

    return avg_precision, coverage

# =============================================================================
# 3. AKILLI ÖNERİ MOTORU (DIVERSITY & ANTI-SERIES)
# =============================================================================
def get_smart_recs(user_id, model, ratings_df, books_dict, n=7):
    """Seri engelleme ve yazar çeşitliliği ile öneri yapar."""
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    read_ids = set(user_history['book_id'])

    # Kullanıcının okuduğu kitapların serilerini bul (title alanındaki seri bilgisini kullan)
    read_series = set()
    for bid in read_ids:
        if bid in books_dict:
            title = books_dict[bid].get('title_full', books_dict[bid]['title'])
            series = extract_series_name(title)
            if series:
                read_series.add(series.lower())

    all_ids = list(set(ratings_df['book_id'].unique()) - read_ids)
    np.random.seed(int(time.time())) # Her seferinde farklı sonuçlar için
    candidates = np.random.choice(all_ids, min(5000, len(all_ids)), replace=False)

    user_preds = []
    for bid in candidates:
        user_preds.append((bid, model.predict(user_id, bid).est))

    user_preds.sort(key=lambda x: x[1], reverse=True)

    final_list = []
    seen_authors = set()
    for i in range(20):
        print()
    print(f"\n👤 KULLANICI {user_id} İÇİN KEŞİF ODAKLI ÖNERİLER:")
    print("-" * 85)
    print(f"{'TAHMİN':<7} | {'YAZAR':<25} | {'KİTAP ADI'}")
    print("-" * 85)

    for bid, score in user_preds:
        if len(final_list) >= n: break

        info = books_dict.get(bid)
        if not info: continue

        author = info['authors']
        title = info['original_title'] if pd.notna(info['original_title']) else info['title']

        if author in seen_authors: continue

        # Aday kitabın serisini kontrol et — okunan serilerden birindeyse atla
        candidate_title = info.get('title_full', info['title'])
        candidate_series = extract_series_name(candidate_title)
        if candidate_series and candidate_series.lower() in read_series:
            continue

        final_list.append(bid)
        seen_authors.add(author)

        d_auth = (author[:23] + '..') if len(author) > 23 else author
        print(f"{score:.2f}/5.0 | {d_auth:<25} | {title}")

    print("-" * 85)
    for i in range(20):
        print()

# =============================================================================
# 5. SERİ ADI ÇIKARMA YARDIMCI FONKSİYONU
# =============================================================================
def extract_series_name(title):
    """Kitap başlığından seri adını çıkarır. Örn: 'Harry Potter and the ... (Harry Potter, #1)' -> 'Harry Potter'"""
    if pd.isna(title):
        return None
    match = re.search(r'\(([^,]+),\s*#\d+', str(title))
    if match:
        return match.group(1).strip()
    return None

# =============================================================================
# 6. AYNI YAZARDAN ÖNERİLER (ML YOK)
# =============================================================================
def get_same_author_recs(user_id, ratings_df, books_df, books_dict, n=5):
    """Kullanıcının okuduğu kitapların yazarlarından başka kitaplar önerir."""
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    read_ids = set(user_history['book_id'])

    # Kullanıcının okuduğu yazarları bul
    read_authors = set()
    for bid in read_ids:
        if bid in books_dict:
            author = books_dict[bid]['authors']
            if pd.notna(author):
                read_authors.add(author)

    if not read_authors:
        return

    # Aynı yazarlardan okunmamış kitapları bul
    same_author_books = books_df[
        (books_df['authors'].isin(read_authors)) &
        (~books_df['book_id'].isin(read_ids))
    ].copy()

    if same_author_books.empty:
        return

    # En popüler kitapları öner (ratings_count'a göre sırala)
    same_author_books = same_author_books.sort_values('ratings_count', ascending=False).head(n)

    print(f"\n📚 BEĞENDİĞİNİZ YAZARLARIN DİĞER POPÜLER KİTAPLARI:")
    print("-" * 85)
    print(f"{'POPÜLERLIK':<12} | {'YAZAR':<25} | {'KİTAP ADI'}")
    print("-" * 85)

    for _, row in same_author_books.iterrows():
        author = str(row['authors'])
        title = row['original_title'] if pd.notna(row['original_title']) else row['title']
        rating_count = row['ratings_count']

        d_auth = (author[:23] + '..') if len(author) > 23 else author
        print(f"{rating_count:<12} | {d_auth:<25} | {title}")

    print("-" * 85)

# =============================================================================
# 7. AYNI SERİDEN ÖNERİLER (ML YOK)
# =============================================================================
def get_series_recs(user_id, ratings_df, books_df, books_dict, n=5):
    """Kullanıcının okuduğu serilerin devam kitaplarını önerir."""
    user_history = ratings_df[ratings_df['user_id'] == user_id]
    read_ids = set(user_history['book_id'])

    # Kullanıcının okuduğu kitapların serilerini bul + okunan kitapları seriye göre grupla
    read_series = {}  # {seri_adı: [okunan kitap başlıkları]}
    for bid in read_ids:
        if bid in books_dict:
            title_full = books_dict[bid].get('title_full', books_dict[bid]['title'])
            series = extract_series_name(title_full)
            if series:
                original = books_dict[bid]['original_title'] if pd.notna(books_dict[bid]['original_title']) else books_dict[bid]['title']
                if series not in read_series:
                    read_series[series] = []
                read_series[series].append(original)

    if not read_series:
        return

    # Aynı seriden okunmamış kitapları seriye göre grupla
    unread_by_series = {}  # {seri_adı: [{kitap bilgisi}]}
    for _, row in books_df.iterrows():
        if row['book_id'] in read_ids:
            continue
        title = row['title']
        series = extract_series_name(title)
        if series and series in read_series:
            original = row['original_title'] if pd.notna(row['original_title']) else title
            if series not in unread_by_series:
                unread_by_series[series] = []
            unread_by_series[series].append({
                'title': original,
                'author': row['authors'],
            })

    if not unread_by_series:
        return

    print(f"\n🔗 OKUDUĞUNUZ SERİLERİN DEVAMI:")
    print("-" * 85)

    count = 0
    for series_name, unread_books in unread_by_series.items():
        if count >= n:
            break

        # Önce okunan kitapları göster
        print(f"\n  📗 \"{series_name}\" SERİSİ — OKUDUKLARINIZ:")
        for read_title in read_series[series_name]:
            print(f"     ✅ {read_title}")
        print()
        # Sonra okunmamış kitapları göster
        print(f"  📖 OKUMADIKLARINIZ:")
        remaining = n - count
        for rec in unread_books[:remaining]:
            author = str(rec['author'])
            d_auth = (author[:23] + '..') if len(author) > 23 else author
            print(f"     ➡️  {rec['title']}  ({d_auth})")
            count += 1
            if count >= n:
                break
        print()

    print("-" * 85)
    

# =============================================================================
# 8. TEK KİTABA GÖRE BENZERLİK ÖNERİSİ (HİBRİT: SVD + TAG İÇERİK)
# =============================================================================
def compute_tag_similarity(book_id, tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name, top_n=50):
    """Bir kitap ile tüm diğer kitaplar arasında tag-tabanlı kosinüs benzerliği hesaplar."""
    if book_id not in book_to_idx:
        return {}, []

    idx = book_to_idx[book_id]
    target_vec = tfidf_matrix[idx]

    # Kosinüs benzerliği (matris zaten L2-normalize edilmiş)
    sims = tfidf_matrix.dot(target_vec.T).toarray().flatten()
    sims[idx] = 0  # kendisiyle benzerliği sıfırla

    # Hedef kitabın en önemli tag'leri
    target_dense = target_vec.toarray().flatten()
    top_tag_indices = np.argsort(target_dense)[::-1][:10]
    target_tags = [idx_to_tag_name.get(i, '?') for i in top_tag_indices if target_dense[i] > 0]

    sim_dict = {}
    for i, bid in enumerate(unique_books):
        if sims[i] > 0:
            sim_dict[bid] = sims[i]

    return sim_dict, target_tags

def get_shared_tags(book_id_a, book_id_b, tfidf_matrix, book_to_idx, idx_to_tag_name, top_n=5):
    """İki kitap arasındaki ortak en önemli tag'leri döndürür."""
    if book_id_a not in book_to_idx or book_id_b not in book_to_idx:
        return []
    vec_a = tfidf_matrix[book_to_idx[book_id_a]].toarray().flatten()
    vec_b = tfidf_matrix[book_to_idx[book_id_b]].toarray().flatten()
    # İki vektörün minimum'u (ortak gücü)
    overlap = np.minimum(vec_a, vec_b)
    top_indices = np.argsort(overlap)[::-1][:top_n]
    return [idx_to_tag_name.get(i, '?') for i in top_indices if overlap[i] > 0]

def get_similar_books(book_id, model, books_dict, ratings_df,
                     tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name, user_id=None, n=7):
    """Hibrit benzerlik: SVD (collab) + Tag (içerik) + hafif popülerlik katkısı."""
    trainset = model.trainset
    alpha = 0.5  # SVD ağırlığı (1-alpha = tag ağırlığı)

    # Eğer kullanıcı ID verilmişse okuduğu kitapları bul
    read_ids = set()
    if user_id is not None:
        user_history = ratings_df[ratings_df['user_id'] == user_id]
        read_ids = set(user_history['book_id'])

    # Kitabın trainset'teki iç ID'sini bul
    try:
        inner_id = trainset.to_inner_iid(book_id)
    except ValueError:
        print(f"\n⚠️ Kitap ID {book_id} eğitim setinde bulunamadı.")
        return

    # --- SVD benzerlik ---
    target_vec = model.qi[inner_id]
    svd_sims = {}
    for other_inner_id in range(trainset.n_items):
        if other_inner_id == inner_id:
            continue
        other_vec = model.qi[other_inner_id]
        cos_sim = np.dot(target_vec, other_vec) / (norm(target_vec) * norm(other_vec) + 1e-9)
        raw_id = trainset.to_raw_iid(other_inner_id)
        svd_sims[raw_id] = cos_sim

    # --- Tag benzerlik ---
    tag_sims, target_tags = compute_tag_similarity(
        book_id, tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name
    )

    # --- Popülerlik katkısı (hafif, baskın değil) ---
    # log(rating_count) ile normalize: en popüler kitap max %5 ek puan alır
    book_rating_counts = ratings_df['book_id'].value_counts().to_dict()
    max_log_count = np.log1p(max(book_rating_counts.values())) if book_rating_counts else 1.0

    # --- Hibrit skor ---
    all_candidates = set(svd_sims.keys()) | set(tag_sims.keys())
    hybrid_scores = []
    for cid in all_candidates:
        if cid in read_ids:  # Okunmuş kitapları atla
            continue
            
        s_svd = svd_sims.get(cid, 0.0)
        s_tag = tag_sims.get(cid, 0.0)
        blended = alpha * s_svd + (1 - alpha) * s_tag

        # Hafif popülerlik katkısı: max +0.05 (en popüler kitaba)
        pop_count = book_rating_counts.get(cid, 0)
        pop_boost = 0.05 * (np.log1p(pop_count) / max_log_count) if max_log_count > 0 else 0
        final = blended + pop_boost

        hybrid_scores.append((cid, final, s_svd, s_tag, pop_boost))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)

    # Hedef kitap bilgisi
    target_info = books_dict.get(book_id, {})
    target_title = target_info.get('original_title') or target_info.get('title', f'ID:{book_id}')
    target_author = target_info.get('authors', 'Bilinmiyor')

    print(f"\n📖 '{target_title}' ({target_author}) KİTABINA BENZER ÖNERİLER (Hibrit):")
    if target_tags:
        print(f"   🏷️  Hedef Kitap Tag'leri: {', '.join(target_tags[:8])}")
    print("-" * 110)
    print(f"{'SKOR':<7} | {'SVD':<6} | {'TAG':<6} | {'POP':<5} | {'YAZAR':<22} | {'KİTAP ADI':<30} | ORTAK TAG'LER")
    print("-" * 110)

    shown = 0
    seen_authors = set()  # Aynı yazardan birden fazla öneri engelle
    for cid, final, s_svd, s_tag, pop_b in hybrid_scores:
        if shown >= n:
            break

        info = books_dict.get(cid)
        if not info:
            continue

        author = info['authors']
        if author in seen_authors:
            continue
        seen_authors.add(author)

        title = info['original_title'] if pd.notna(info['original_title']) else info['title']

        # Ortak tag'leri bul
        shared = get_shared_tags(book_id, cid, tfidf_matrix, book_to_idx, idx_to_tag_name, top_n=4)
        shared_str = ', '.join(shared) if shared else '-'

        d_auth = (author[:20] + '..') if len(author) > 20 else author
        d_title = (str(title)[:28] + '..') if len(str(title)) > 28 else str(title)
        print(f"{final:.4f}  | {s_svd:.3f}  | {s_tag:.3f}  | +{pop_b:.3f} | {d_auth:<22} | {d_title:<30} | {shared_str}")
        shown += 1

    print("-" * 110)

# =============================================================================
# 8b. BENZERLİK FONKSİYONU GEÇERLİLİK METRİKLERİ
# =============================================================================
def validate_similarity(model, books_dict, ratings_df,
                       tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name,
                       sample_size=100, k=10):
    """Hibrit benzerlik fonksiyonunun kalitesini ölçen metrikler:
    1. Precision@k (Aynı Yazar Testi): Aynı yazarın kitapları top-k'da çıkıyor mu?
    2. Tag Overlap (Jaccard): Önerilen kitapların tag'leri hedef kitapla ne kadar örtüşüyor?
    3. Popülerlik Dağılımı: Sonuçlar sadece popüler kitaplardan mı oluşuyor?
    """
    print("\n📐 Benzerlik Fonksiyonu Geçerlilik Testi Başlıyor...")
    trainset = model.trainset
    alpha = 0.5

    book_rating_counts = ratings_df['book_id'].value_counts().to_dict()
    max_log_count = np.log1p(max(book_rating_counts.values())) if book_rating_counts else 1.0

    # Örneklem seç
    all_inner_ids = list(range(trainset.n_items))
    np.random.seed(42)
    sample_ids = np.random.choice(all_inner_ids, min(sample_size, len(all_inner_ids)), replace=False)

    author_precision_scores = []
    tag_jaccard_scores = []
    rec_pop_counts = []
    all_pop_counts = list(book_rating_counts.values())

    for inner_id in sample_ids:
        raw_id = trainset.to_raw_iid(inner_id)
        target_info = books_dict.get(raw_id)
        if not target_info:
            continue
        target_author = target_info.get('authors', '')

        # SVD benzerlik
        target_vec = model.qi[inner_id]
        svd_sims = {}
        for oid in range(trainset.n_items):
            if oid == inner_id:
                continue
            other_vec = model.qi[oid]
            cos_sim = np.dot(target_vec, other_vec) / (norm(target_vec) * norm(other_vec) + 1e-9)
            svd_sims[trainset.to_raw_iid(oid)] = cos_sim

        # Tag benzerlik
        tag_sims_dict, _ = compute_tag_similarity(
            raw_id, tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name
        )

        # Hibrit skor
        all_cands = set(svd_sims.keys()) | set(tag_sims_dict.keys())
        scored = []
        for cid in all_cands:
            s_svd = svd_sims.get(cid, 0.0)
            s_tag = tag_sims_dict.get(cid, 0.0)
            blended = alpha * s_svd + (1 - alpha) * s_tag
            pop_count = book_rating_counts.get(cid, 0)
            pop_boost = 0.05 * (np.log1p(pop_count) / max_log_count) if max_log_count > 0 else 0
            scored.append((cid, blended + pop_boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [s[0] for s in scored[:k]]

        # 1. Aynı Yazar Precision
        if target_author:
            same_author_in_topk = sum(
                1 for bid in top_k_ids
                if books_dict.get(bid, {}).get('authors', '') == target_author
            )
            # Toplam aynı yazar kitap sayısı (üst sınır)
            total_same_author = sum(
                1 for bid in books_dict
                if books_dict[bid].get('authors', '') == target_author and bid != raw_id
            )
            if total_same_author > 0:
                author_precision_scores.append(same_author_in_topk / min(total_same_author, k))

        # 2. Tag Jaccard
        if raw_id in book_to_idx:
            target_tag_vec = (tfidf_matrix[book_to_idx[raw_id]].toarray().flatten() > 0)
            jaccards = []
            for bid in top_k_ids:
                if bid in book_to_idx:
                    rec_tag_vec = (tfidf_matrix[book_to_idx[bid]].toarray().flatten() > 0)
                    intersection = np.sum(target_tag_vec & rec_tag_vec)
                    union = np.sum(target_tag_vec | rec_tag_vec)
                    if union > 0:
                        jaccards.append(intersection / union)
            if jaccards:
                tag_jaccard_scores.append(np.mean(jaccards))

        # 3. Popülerlik
        for bid in top_k_ids:
            rec_pop_counts.append(book_rating_counts.get(bid, 0))

    # --- Raporla ---
    print("\n" + "=" * 60)
    print("📐 BENZERLİK FONKSİYONU GEÇERLİLİK RAPORU")
    print("=" * 60)

    if author_precision_scores:
        avg_ap = np.mean(author_precision_scores)
        print(f"✍️  Aynı Yazar Precision@{k} : {avg_ap:.4f}")
        print(f"   (Aynı yazarın kitapları top-{k}'da ne oranda çıkıyor)")
    else:
        print("✍️  Aynı Yazar Precision@k   : Hesaplanamadı")

    if tag_jaccard_scores:
        avg_jac = np.mean(tag_jaccard_scores)
        print(f"🏷️  Ort. Tag Jaccard@{k}     : {avg_jac:.4f}")
        print(f"   (Önerilen kitapların tag benzerliği, 1.0 = mükemmel)")
    else:
        print(f"🏷️  Ort. Tag Jaccard@{k}     : Hesaplanamadı")

    if rec_pop_counts:
        avg_rec_pop = np.mean(rec_pop_counts)
        median_rec_pop = np.median(rec_pop_counts)
        avg_all_pop = np.mean(all_pop_counts)
        median_all_pop = np.median(all_pop_counts)
        print(f"📊 Önerilen Kitapların Popülerliği:")
        print(f"   Önerilenlerin Ortalaması  : {avg_rec_pop:.0f}  (tüm katalog: {avg_all_pop:.0f})")
        print(f"   Önerilenlerin Medyanı    : {median_rec_pop:.0f}  (tüm katalog: {median_all_pop:.0f})")
        ratio = avg_rec_pop / avg_all_pop if avg_all_pop > 0 else 0
        if ratio > 3:
            print(f"   ⚠️ Öneriler katalog ortalamasının {ratio:.1f}x popüler — popülerlik baskın olabilir")
        elif ratio > 1.5:
            print(f"   ✅ Öneriler biraz daha popüler ({ratio:.1f}x) — dengeli")
        else:
            print(f"   ✅ Popülerlik dengeli ({ratio:.1f}x)")

    print("=" * 60)

# =============================================================================
# 9. ANA ÇALIŞTIRMA GÖVDESİ
# =============================================================================
def main():
    # Veri yükleme
    books_df, ratings_df, books_dict = load_data()

    # books_dict'e full title bilgisini ekle (seri tespiti için)
    title_map = books_df.set_index('book_id')['title'].to_dict()
    for bid in books_dict:
        books_dict[bid]['title_full'] = title_map.get(bid, '')

    # Tag verileri yükle ve profil matrisi oluştur
    book_tags_df = load_tags(books_df)
    tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name = build_tag_profiles(book_tags_df)

    # Surprise Dataset Hazırlığı
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)

    # Eğitim ve Test setini ayır
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # SVD Modeli Yapılandırması
    print("\n🚀 SVD (Matrix Factorization) Modeli Eğitiliyor...")
    start_time = time.time()
    # Optimize edilmiş hiperparametreler: daha fazla faktör + düşük regularizasyon
    model = SVD(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.08, random_state=42)
    model.fit(trainset)
    train_time = time.time() - start_time

    # --- EZBERLEME TESTİ (RMSE) ---
    # Eğitim RMSE: Modelin gördüğü verideki başarısı
    train_preds = model.test(trainset.build_testset())
    train_rmse = accuracy.rmse(train_preds, verbose=False)

    # Test RMSE: Modelin hiç görmediği verideki başarısı
    test_preds = model.test(testset)
    test_rmse = accuracy.rmse(test_preds, verbose=False)

    # Metrikleri Hesapla
    p_at_5, coverage = calculate_metrics(model, ratings_df, k=5)

    # =========================================================================
    # ÇIKTI RAPORU
    # =========================================================================
    print("\n" + "="*60)
    print("📊 MODEL ÖĞRENME ANALİZ RAPORU")
    print("="*60)
    print(f"📉 Eğitim (Train) RMSE : {train_rmse:.4f} (Ezber Gücü)")
    print(f"📉 Test (Validation) RMSE: {test_rmse:.4f} (Öğrenme Gücü)")

    # Analiz: RMSE farkı modelin genelleme yeteneğini gösterir
    gap = test_rmse - train_rmse
    if gap > 0.15:
        print("⚠️ DURUM: Hafif Ezberleme (Overfitting) var.")
    else:
        print("✅ DURUM: Sağlıklı Öğrenme (Generalization). Model genelleme yapabiliyor.")

    print("-" * 60)
    print(f"📈 Precision@5 : %{p_at_5 * 100:.2f}")
    print(f"🌐 Katalog Kapsamı : %{coverage:.2f}")
    print(f"⏱️ Eğitim Süresi : {train_time:.2f} saniye")
    print("="*60)

    # Önerileri Göster
    sample_uid = 9 #ratings_df['user_id'].iloc[0]
    get_smart_recs(sample_uid, model, ratings_df, books_dict)
    get_same_author_recs(sample_uid, ratings_df, books_df, books_dict)
    get_series_recs(sample_uid, ratings_df, books_df, books_dict)

    # Tek kitaba göre benzerlik önerisi (Hibrit: SVD + Tag)
    sample_book_id = 1  # The Hunger Games
    get_similar_books(sample_book_id, model, books_dict, ratings_df,
                      tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name, user_id=sample_uid)

    # Benzerlik Fonksiyonu Geçerlilik Metrikleri
    validate_similarity(model, books_dict, ratings_df,
                        tfidf_matrix, book_to_idx, unique_books, idx_to_tag_name)

if __name__ == "__main__":
    main()