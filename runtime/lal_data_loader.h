#ifndef LAL_DATA_LOADER_H
#define LAL_DATA_LOADER_H
/*
 * lal_data_loader.h - LAL binary training-data loader.
 *
 * Reads the .bin files produced by scripts/tokenize_data.py and provides
 * random-access / random-batch access to tokenized samples for training.
 *
 * Binary file format (little-endian), written by tokenize_data.py:
 *     [magic     : 4 bytes  = "LALT"]
 *     [n_samples : int32]
 *     [n_vocab   : int32]
 *     then n_samples records of:
 *         [n_tokens  : int32]
 *         [token_ids : int32 * n_tokens]
 *
 * This is a single-header library. In exactly ONE .c translation unit define
 * the implementation macro BEFORE including this header to get the function
 * definitions; other files include it normally for the declarations:
 *
 *     #define LAL_DATA_LOADER_IMPLEMENTATION
 *     #include "lal_data_loader.h"
 *
 * A ready-to-compile self test is provided behind _LAL_DATA_LOADER_TEST (see
 * the bottom of this file) so the header can be verified independently:
 *
 *     cc -std=c99 -D_LAL_DATA_LOADER_TEST=1 runtime/lal_data_loader.h -o loader_test
 *     ./loader_test data/train_tokens.bin
 *
 * NOTE: `offsets` stores byte offsets as `int`, which is sufficient for data
 * files up to ~2 GB. For larger files change it to `long`/`int64_t`.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAL_DATA_MAGIC     "LALT"   /* 4-byte magic (compared with memcmp) */
#define LAL_DATA_MAGIC_LEN 4

/* A single training sample: a token count plus a heap-allocated token array. */
typedef struct {
    int  n_tokens;
    int *tokens;
} TrainSample;

/*
 * Random-access handle over a LAL binary data file.
 *   f         : open file handle (kept open for random access)
 *   n_samples : number of samples in the file
 *   offsets   : byte offset of each sample's [n_tokens] field
 *   max_len   : max tokens returned/stored per sample (caller may set)
 *   n_vocab   : vocabulary size recorded in the file header
 */
typedef struct {
    FILE *f;
    int   n_samples;
    int  *offsets;
    int   max_len;
    int   n_vocab;
} DataLoader;

/* ---- API ---- */

/* Open the binary file, validate the header, and build a per-sample byte
 * offset index so any sample can be fetched with a single fseek/fread.
 * dl->max_len defaults to 512; override after init if you need another cap.
 * Returns 0 on success, non-zero on error (dl is safe to pass to free). */
int  dataloader_init(DataLoader *dl, const char *path);

/* Fetch sample idx into the caller buffer `tokens` (capacity max_len).
 * Tokens beyond max_len are dropped. Returns the number of tokens written
 * (<= max_len), or a negative value on error. */
int  dataloader_get(DataLoader *dl, int idx, int *tokens, int max_len);

/* Fetch a batch of samples given an array of (already-shuffled) indices.
 *
 * `tokens` is a CALLER-ALLOCATED array of `batch_size` int* slots
 * (e.g. `int **t = malloc(batch_size * sizeof(int*));`). For each slot the
 * function allocates a row of `dl->max_len` ints and fills it with the
 * sample's token ids; lengths[i] receives that row's token count.
 *
 * Free the rows with dataloader_free_batch(); the caller frees the outer
 * `tokens` array itself.
 *
 * Returns batch_size on success, or a negative value on error (rows already
 * allocated are freed before returning). */
int  dataloader_random_batch(DataLoader *dl, int *indices, int batch_size,
                             int **tokens, int *lengths);

/* Free the rows allocated by dataloader_random_batch. Does NOT free the
 * outer `tokens` array (the caller owns it). */
void dataloader_free_batch(int **tokens, int batch_size);

/* Release all resources held by the loader. Safe on a zeroed struct. */
void dataloader_free(DataLoader *dl);

/* =========================================================================
 * Implementation
 *
 * Compiled in when LAL_DATA_LOADER_IMPLEMENTATION is defined, or implicitly
 * when the self-test (_LAL_DATA_LOADER_TEST) is enabled. Everything lives
 * inside the include guard, so multiple #include in one TU is safe.
 * ========================================================================= */
#if defined(LAL_DATA_LOADER_IMPLEMENTATION) || defined(_LAL_DATA_LOADER_TEST)

#include <errno.h>

int dataloader_init(DataLoader *dl, const char *path) {
    if (!dl || !path) return 1;
    memset(dl, 0, sizeof(*dl));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[dataloader] cannot open '%s': %s\n",
                path, strerror(errno));
        return 2;
    }

    /* ---- header ---- */
    char magic[LAL_DATA_MAGIC_LEN];
    if (fread(magic, 1, LAL_DATA_MAGIC_LEN, f) != LAL_DATA_MAGIC_LEN ||
        memcmp(magic, LAL_DATA_MAGIC, LAL_DATA_MAGIC_LEN) != 0) {
        fprintf(stderr, "[dataloader] bad magic in '%s' (expected \"LALT\")\n", path);
        fclose(f);
        return 3;
    }

    int32_t n_samples = 0, n_vocab = 0;
    if (fread(&n_samples, sizeof(int32_t), 1, f) != 1 ||
        fread(&n_vocab,   sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "[dataloader] truncated header in '%s'\n", path);
        fclose(f);
        return 4;
    }

    dl->f         = f;
    dl->n_samples = (int)n_samples;
    dl->n_vocab   = (int)n_vocab;
    dl->max_len   = 512;
    dl->offsets   = NULL;

    if (dl->n_samples <= 0) {
        /* valid but empty file; offsets stays NULL */
        return 0;
    }

    /* ---- build per-sample byte offset index ---- */
    dl->offsets = (int *)malloc(sizeof(int) * (size_t)dl->n_samples);
    if (!dl->offsets) {
        fprintf(stderr, "[dataloader] OOM allocating offsets (%d)\n", dl->n_samples);
        fclose(f);
        dl->f = NULL;
        return 5;
    }

    long cur = ftell(f); /* first sample begins right after the header */
    for (int i = 0; i < dl->n_samples; i++) {
        dl->offsets[i] = (int)cur;
        int32_t n_tok = 0;
        if (fread(&n_tok, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "[dataloader] truncated at sample %d (reading n_tokens)\n", i);
            free(dl->offsets);
            dl->offsets = NULL;
            fclose(f);
            dl->f = NULL;
            return 6;
        }
        if (n_tok < 0) {
            fprintf(stderr, "[dataloader] negative n_tokens=%d at sample %d\n", n_tok, i);
            free(dl->offsets);
            dl->offsets = NULL;
            fclose(f);
            dl->f = NULL;
            return 7;
        }
        /* skip over the token id payload */
        if (n_tok > 0) {
            if (fseek(f, (long)sizeof(int32_t) * n_tok, SEEK_CUR) != 0) {
                fprintf(stderr, "[dataloader] seek failed at sample %d\n", i);
                free(dl->offsets);
                dl->offsets = NULL;
                fclose(f);
                dl->f = NULL;
                return 8;
            }
        }
        cur += (long)sizeof(int32_t) * (1 + n_tok);
    }

    return 0;
}

int dataloader_get(DataLoader *dl, int idx, int *tokens, int max_len) {
    if (!dl || !tokens)                 return -1;
    if (!dl->f || !dl->offsets)         return -2;
    if (idx < 0 || idx >= dl->n_samples) return -3;
    if (max_len <= 0)                   return -4;

    if (fseek(dl->f, (long)dl->offsets[idx], SEEK_SET) != 0) return -5;

    int32_t n_tok = 0;
    if (fread(&n_tok, sizeof(int32_t), 1, dl->f) != 1) return -6;

    int to_read = (n_tok > max_len) ? max_len : (int)n_tok;
    if (to_read > 0) {
        if (fread(tokens, sizeof(int32_t), (size_t)to_read, dl->f)
            != (size_t)to_read) {
            return -7;
        }
    }
    /* remember the largest cap seen, for batch allocation convenience */
    if (max_len > dl->max_len) dl->max_len = max_len;
    return to_read;
}

int dataloader_random_batch(DataLoader *dl, int *indices, int batch_size,
                            int **tokens, int *lengths) {
    if (!dl || !tokens || !lengths) return -1;
    if (batch_size <= 0)            return -2;
    if (!indices)                   return -3;

    int max_len = dl->max_len > 0 ? dl->max_len : 512;

    for (int i = 0; i < batch_size; i++) {
        tokens[i] = (int *)malloc(sizeof(int) * (size_t)max_len);
        if (!tokens[i]) {
            /* free every row allocated so far, then bail */
            for (int j = 0; j < i; j++) { free(tokens[j]); tokens[j] = NULL; }
            return -(10 + i);
        }
        int n = dataloader_get(dl, indices[i], tokens[i], max_len);
        if (n < 0) {
            free(tokens[i]); tokens[i] = NULL;
            for (int j = 0; j < i; j++) { free(tokens[j]); tokens[j] = NULL; }
            return -(100 + i);
        }
        lengths[i] = n;
    }
    return batch_size;
}

void dataloader_free_batch(int **tokens, int batch_size) {
    if (!tokens || batch_size <= 0) return;
    for (int i = 0; i < batch_size; i++) {
        if (tokens[i]) { free(tokens[i]); tokens[i] = NULL; }
    }
    /* NOTE: does not free `tokens` itself; the caller owns the outer array. */
}

void dataloader_free(DataLoader *dl) {
    if (!dl) return;
    if (dl->f) { fclose(dl->f); dl->f = NULL; }
    if (dl->offsets) { free(dl->offsets); dl->offsets = NULL; }
    dl->n_samples = 0;
    dl->n_vocab   = 0;
}

#endif /* LAL_DATA_LOADER_IMPLEMENTATION || _LAL_DATA_LOADER_TEST */

/* =========================================================================
 * Self test — compile with -D_LAL_DATA_LOADER_TEST=1 to enable main().
 * ========================================================================= */
#ifdef _LAL_DATA_LOADER_TEST

#include <time.h>

/* simple Fisher-Yates shuffle on a freshly allocated index array */
static void lal_dl_shuffle(int *a, int n, unsigned int seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "data/train_tokens.bin";
    printf("[test] opening %s\n", path);

    DataLoader dl;
    int rc = dataloader_init(&dl, path);
    if (rc != 0) {
        printf("[test] dataloader_init failed: %d\n", rc);
        return 1;
    }
    printf("[test] n_samples=%d  n_vocab=%d  max_len=%d\n",
           dl.n_samples, dl.n_vocab, dl.max_len);

    /* show the first sample */
    int *buf = (int *)malloc(sizeof(int) * (size_t)dl.max_len);
    int n0 = dataloader_get(&dl, 0, buf, dl.max_len);
    printf("[test] sample[0]: n_tokens=%d  ids[0:8]=", n0);
    for (int i = 0; i < (n0 < 8 ? n0 : 8); i++) printf("%d ", buf[i]);
    printf("\n");
    free(buf);

    /* random batch of 4 (caller allocates the outer row-pointer array) */
    int batch = 4;
    int *indices = (int *)malloc(sizeof(int) * (size_t)dl.n_samples);
    for (int i = 0; i < dl.n_samples; i++) indices[i] = i;
    lal_dl_shuffle(indices, dl.n_samples, (unsigned)time(NULL));

    int **b_tokens = (int **)malloc(sizeof(int *) * (size_t)batch);
    int  *b_lens   = (int  *)malloc(sizeof(int)   * (size_t)batch);
    int br = dataloader_random_batch(&dl, indices, batch, b_tokens, b_lens);
    printf("[test] random_batch returned %d\n", br);
    if (br == batch) {
        for (int i = 0; i < batch; i++) {
            printf("[test]   batch[%d] idx=%d len=%d ids[0:5]=",
                   i, indices[i], b_lens[i]);
            int show = b_lens[i] < 5 ? b_lens[i] : 5;
            for (int k = 0; k < show; k++) printf("%d ", b_tokens[i][k]);
            printf("\n");
        }
    }
    dataloader_free_batch(b_tokens, batch);  /* frees the rows */
    free(b_tokens);                          /* caller frees the outer array */
    free(b_lens);
    free(indices);

    dataloader_free(&dl);
    printf("[test] OK\n");
    return 0;
}

#endif /* _LAL_DATA_LOADER_TEST */

#endif /* LAL_DATA_LOADER_H */
