#include <stdint.h>
#include <stddef.h>

#define CELLS 65536u
#define MAX_TRITS 1024u
#define ROUTE_CAP 64u

typedef enum {
    OK = 0,
    ERR_MEMORY = 1,
    ERR_RANGE = 2,
    ERR_EMPTY = 3,
    ERR_FULL = 4
} Status;

typedef struct {
    int8_t *digits;
    uint32_t length;
} Ternary;

typedef struct {
    uint32_t source;
    uint32_t destination;
    uint32_t predicate;
    uint32_t payload;
} Message;

typedef struct {
    Message entries[ROUTE_CAP];
    uint32_t read;
    uint32_t write;
    uint32_t count;
} Ring;

typedef struct {
    uint8_t p[3];
} Permutation;

static int8_t arena[CELLS];
static uint32_t cursor = 0;

static Status allocate_trits(
    uint32_t count,
    int8_t **out
) {
    if (out == NULL) return ERR_RANGE;
    if (count > CELLS - cursor)
        return ERR_MEMORY;

    *out = &arena[cursor];
    cursor += count;
    return OK;
}

static void normalize(Ternary *x) {
    while (x->length > 1 &&
           x->digits[x->length - 1] == 0) {
        x->length--;
    }
}

static Status ternary_zero(Ternary *x) {
    if (x == NULL) return ERR_RANGE;

    Status s = allocate_trits(1, &x->digits);
    if (s != OK) return s;

    x->digits[0] = 0;
    x->length = 1;
    return OK;
}

static int32_t floor_div3(int32_t x) {
    int32_t q = x / 3;
    int32_t r = x % 3;
    if (r < 0) q--;
    return q;
}

static Status ternary_from_i64(
    int64_t value,
    Ternary *out
) {
    if (out == NULL) return ERR_RANGE;

    int8_t tmp[64];
    uint32_t n = 0;

    if (value == 0)
        return ternary_zero(out);

    while (value != 0) {
        int64_t q = value / 3;
        int64_t r = value % 3;

        if (r == 2) {
            r = -1;
            q++;
        } else if (r == -2) {
            r = 1;
            q--;
        }

        if (n >= 64) return ERR_RANGE;
        tmp[n++] = (int8_t)r;
        value = q;
    }

    Status s = allocate_trits(n, &out->digits);
    if (s != OK) return s;

    out->length = n;
    for (uint32_t i = 0; i < n; i++)
        out->digits[i] = tmp[i];

    return OK;
}

/* Decoding oracle: evaluate balanced ternary to int64.
 * Returns ERR_RANGE if any digit is out of {-1,0,1} or the value
 * does not fit in int64. */
static Status ternary_to_i64(
    const Ternary *x,
    int64_t *out
) {
    if (!x || !out || !x->digits) return ERR_RANGE;

    int64_t value = 0;
    int64_t place = 1;

    for (uint32_t i = 0; i < x->length; i++) {
        int8_t d = x->digits[i];
        if (d < -1 || d > 1) return ERR_RANGE;

        /* Detect overflow before multiplying place further. */
        if (i > 0 && place > INT64_MAX / 3) return ERR_RANGE;

        value += (int64_t)d * place;
        if (i + 1 < x->length)
            place *= 3;
    }

    *out = value;
    return OK;
}

static Status ternary_add(
    const Ternary *a,
    const Ternary *b,
    Ternary *out
) {
    if (!a || !b || !out) return ERR_RANGE;
    if (!a->digits || !b->digits)
        return ERR_RANGE;

    uint32_t n = a->length > b->length
               ? a->length : b->length;

    if (n >= MAX_TRITS) return ERR_RANGE;

    int8_t *result;
    Status s = allocate_trits(n + 1, &result);
    if (s != OK) return s;

    int32_t carry = 0;

    for (uint32_t i = 0; i < n; i++) {
        int32_t x = i < a->length
                  ? a->digits[i] : 0;
        int32_t y = i < b->length
                  ? b->digits[i] : 0;

        if (x < -1 || x > 1 ||
            y < -1 || y > 1)
            return ERR_RANGE;

        int32_t z = x + y + carry;
        carry = floor_div3(z + 1);
        result[i] = (int8_t)(z - 3 * carry);
    }

    result[n] = (int8_t)carry;
    out->digits = result;
    out->length = n + 1;
    normalize(out);
    return OK;
}

static Status ternary_mul(
    const Ternary *a,
    const Ternary *b,
    Ternary *out
) {
    if (!a || !b || !out) return ERR_RANGE;
    if (!a->digits || !b->digits)
        return ERR_RANGE;

    if (a->length == 0 || b->length == 0 ||
        a->length > MAX_TRITS ||
        b->length > MAX_TRITS)
        return ERR_RANGE;

    uint32_t n = a->length + b->length;
    if (n + 1 > MAX_TRITS) return ERR_RANGE;

    int8_t *result;
    Status s = allocate_trits(n + 1, &result);
    if (s != OK) return s;

    for (uint32_t i = 0; i <= n; i++)
        result[i] = 0;

    for (uint32_t i = 0; i < a->length; i++) {
        if (a->digits[i] < -1 ||
            a->digits[i] > 1)
            return ERR_RANGE;
    }

    for (uint32_t j = 0; j < b->length; j++) {
        if (b->digits[j] < -1 ||
            b->digits[j] > 1)
            return ERR_RANGE;
    }

    int32_t carry = 0;

    for (uint32_t k = 0; k < n; k++) {
        int32_t sum = carry;

        for (uint32_t i = 0; i < a->length; i++) {
            if (k >= i) {
                uint32_t j = k - i;
                if (j < b->length)
                    sum += (int32_t)a->digits[i] *
                           (int32_t)b->digits[j];
            }
        }

        carry = floor_div3(sum + 1);
        result[k] = (int8_t)(sum - 3 * carry);
    }

    if (carry < -1 || carry > 1)
        return ERR_RANGE;

    result[n] = (int8_t)carry;
    out->digits = result;
    out->length = n + 1;
    normalize(out);
    return OK;
}

static Status pack_trits(
    const Ternary *x,
    uint8_t *buffer,
    uint32_t capacity,
    uint32_t *written
) {
    if (!x || !buffer || !written)
        return ERR_RANGE;

    // Five base-3 digits fit into one byte:
    // 3^5 = 243 < 256.
    uint32_t required = (x->length + 4) / 5;

    if (capacity < required) return ERR_MEMORY;

    for (uint32_t k = 0; k < required; k++) {
        uint32_t accumulator = 0;
        uint32_t place = 1;

        for (uint32_t j = 0; j < 5; j++) {
            uint32_t i = k * 5 + j;
            int32_t digit = i < x->length
                          ? x->digits[i] : 0;

            if (digit < -1 || digit > 1)
                return ERR_RANGE;

            accumulator +=
                (uint32_t)(digit + 1) * place;
            place *= 3;
        }

        buffer[k] = (uint8_t)accumulator;
    }

    *written = required;
    return OK;
}

static Status route_push(
    Ring *r,
    Message message
) {
    if (!r) return ERR_RANGE;
    if (r->count >= ROUTE_CAP) return ERR_FULL;

    r->entries[r->write] = message;
    r->write = (r->write + 1) % ROUTE_CAP;
    r->count++;
    return OK;
}

static Status route_pop(
    Ring *r,
    Message *out
) {
    if (!r || !out) return ERR_RANGE;
    if (r->count == 0) return ERR_EMPTY;

    *out = r->entries[r->read];
    r->read = (r->read + 1) % ROUTE_CAP;
    r->count--;
    return OK;
}

static Status permutation_compose(
    const Permutation *a,
    const Permutation *b,
    Permutation *out
) {
    if (!a || !b || !out) return ERR_RANGE;

    uint8_t mask_a = 0;
    uint8_t mask_b = 0;

    for (uint32_t i = 0; i < 3; i++) {
        if (a->p[i] >= 3 || b->p[i] >= 3)
            return ERR_RANGE;

        mask_a |= (uint8_t)(1u << a->p[i]);
        mask_b |= (uint8_t)(1u << b->p[i]);
    }

    if (mask_a != 7 || mask_b != 7)
        return ERR_RANGE;

    Permutation result;

    for (uint32_t i = 0; i < 3; i++)
        result.p[i] = a->p[b->p[i]];

    *out = result;
    return OK;
}

static int permutation_equal(
    const Permutation *a,
    const Permutation *b
) {
    for (uint32_t i = 0; i < 3; i++)
        if (a->p[i] != b->p[i])
            return 0;
    return 1;
}

int main(void) {
    Ternary a, b, sum, product;
    uint8_t encoded[32];
    uint32_t bytes;
    int64_t decoded;

    if (ternary_from_i64(42, &a) != OK)
        return 1;

    if (ternary_from_i64(-17, &b) != OK)
        return 2;

    if (ternary_add(&a, &b, &sum) != OK)
        return 3;

    /* Decoding oracle: 42 + (-17) must equal 25. */
    if (ternary_to_i64(&sum, &decoded) != OK || decoded != 25)
        return 3;

    if (ternary_mul(&a, &b, &product) != OK)
        return 4;

    /* Decoding oracle: 42 * (-17) must equal -714. */
    if (ternary_to_i64(&product, &decoded) != OK || decoded != -714)
        return 4;

    if (pack_trits(&sum, encoded,
                   sizeof(encoded), &bytes) != OK)
        return 5;

    Ring ring = {0};
    Message message = {1, 2, 7, 42};
    Message received;

    if (route_push(&ring, message) != OK)
        return 6;

    if (route_pop(&ring, &received) != OK)
        return 7;

    if (received.payload != 42)
        return 8;

    Permutation p = {{1, 0, 2}};
    Permutation q = {{0, 2, 1}};
    Permutation pq, qp;

    if (permutation_compose(&p, &q, &pq) != OK)
        return 9;

    if (permutation_compose(&q, &p, &qp) != OK)
        return 10;

    /* Verify these permutations do not commute. */
    if (permutation_equal(&pq, &qp))
        return 11;

    return 0;
}
