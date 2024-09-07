/**
 * @file mm.c
 * @brief A 64-bit struct-based implicit free list memory allocator
 *
 * 15-213: Introduction to Computer Systems
 * This allocator implements segregated free lists to categorize free memory
 blocks
 * by size, enhancing allocation efficiency and speed. It incorporates a unique
 strategy
 * for handling small "mini blocks" to reduce fragmentation and utilize space
 more effectively.
 *
 * Removing footers from allocated blocks and footer only in free blocks.
 * Removes prev_free pointer for mini blocks
 *
 * Key Features:
 * - Segregated Free Lists: Organized by size ranges for faster allocation.
 * - Mini Blocks: Special handling for small allocations to minimize space
 waste.
 * - Best-Fit Placement
 *
 *
 * Allocated Block: [ Header | Payload ]
 *
 *
 * Free Block: [ Header | Unused Payload Space | Footer ]

 *************************************************************************
 *
 * ADVICE FOR STUDENTS.
 * - Step 0: Please read the writeup!
 * - Step 1: Write your heap checker.
 * - Step 2: Write contracts / debugging assert statements.
 * - Good luck, and have fun!
 *
 *************************************************************************
 *
 * @author Leila Lei <xlei2@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printf(...) ((void)printf(__VA_ARGS__))
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, these should emit no code whatsoever,
 * not even from evaluation of argument expressions.  However,
 * argument expressions should still be syntax-checked and should
 * count as uses of any variables involved.  This used to use a
 * straightforward hack involving sizeof(), but that can sometimes
 * provoke warnings about misuse of sizeof().  I _hope_ that this
 * newer, less straightforward hack will be more robust.
 * Hat tip to Stack Overflow poster chqrlie (see
 * https://stackoverflow.com/questions/72647780).
 */
#define dbg_discard_expr_(...) ((void)((0) && printf(__VA_ARGS__)))
#define dbg_requires(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_assert(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_ensures(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_printf(...) dbg_discard_expr_(__VA_ARGS__)
#define dbg_printheap(...) ((void)((0) && print_heap(__VA_ARGS__)))
#endif

/* Basic constants */
#define NUM_FREE_LISTS 15

typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/**
 * @brief extension size
 * The increment size for heap expansion when no suitable free blocks are
 * available. Chosen to balance the overhead of frequent heap extensions and
 * excessive memory allocation. Its size ensures alignment and is a multiple of
 * double word size. (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 12);

/**
 * @brief Allocation status mask
 * A mask that can be used to extract or set the allocation status of a block.
 */
static const word_t alloc_mask = 0x1;      // 0001 - Allocation status
static const word_t prev_alloc_mask = 0x2; // 0010 - Previous allocation status
static const word_t mini_mask = 0x4;       // 0100 - Mini-block status
static const word_t prev_mini_mask = 0x8;  // 1000 - Previous mini-block status

/**
 * @brief size mask
 * A mask that can be used to extract the size of a block from its
 * header/footer.
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block {
    /** @brief Header contains size + allocation flag */
    word_t header;

    /**
     * @brief A pointer to the block payload.
     *
     */
    union {
        struct {
            struct block *next_free; // Pointer to the previous free block in
                                     // the free list
            struct block
                *prev_free; // Pointer to the next free block in the free list
        };
        char payload[0];
    };

} block_t;

/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *seg_free_list[NUM_FREE_LISTS];

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *  the allocation status is in the 0th bit, the allocation status of the
 * previous block is in the 1st bit, the mini-block status in the 2nd bit, and
 * the mini-block status of the previous block in the 3rd bit.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @param[in] prev_alloc true if the previous block is allocated, and false if
 * the previous block is free.
 * @param[in] mini true if the block is a mini block, and false otherwise.
 * @param[in] prev_mini whether the previous block in memory is a mini block.
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool prev_alloc, bool mini,
                   bool prev_mini) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (prev_alloc) {
        word |= prev_alloc_mask;
    }
    if (mini) {
        word |= mini_mask;
    }
    if (prev_mini) {
        word |= prev_mini_mask;
    }

    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer an allocated block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 *
 * The header is found by subtracting the block size from
 * the footer and adding back wsize.
 *
 * If the prologue is given, then the footer is return as the block.
 *
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    if (size == 0) {
        return (block_t *)footer;
    }
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

// Extracts the previous allocation status from a word
static bool extract_prev_alloc(word_t word) {
    return (bool)(word & prev_alloc_mask);
}

static bool get_prev_alloc(block_t *block) {
    return extract_prev_alloc(block->header);
}

// Extracts the mini block status from a word
static bool extract_mini(word_t word) {
    return (bool)(word & mini_mask);
}

// getting the mini status from a block
static bool get_mini(block_t *block) {
    return extract_mini(block->header);
}

// Extracts the previous mini block status from a word
static bool extract_prev_mini(word_t word) {
    return (bool)(word & prev_mini_mask);
}

// getting the prev mini status from a block
static bool get_prev_mini(block_t *block) {
    return extract_prev_mini(block->header);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue serves as a marker at the end of the heap, indicating that there
 * are no more blocks beyond it.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == (char *)mem_heap_hi() - 7);
    block->header = pack(0, true, false, false, false);
}

static void update_next_block_header(block_t *next_block, bool prev_alloc,
                                     bool prev_mini) {
    size_t size = get_size(next_block);
    bool alloc = get_alloc(next_block);
    bool mini = get_mini(next_block);
    next_block->header = pack(size, alloc, prev_alloc, mini, prev_mini);
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * locates the next consecutive block in the heap given a starting block.
 * It performs pointer arithmetic based on the size of the current block to find
 * the start of the next block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Writes a block starting at the given address.
 *
 * initializes a block at the specified location with a given size and
 * allocation status, including additional attributes regarding the allocation
 * status of the previous block and whether the block or its predecessor is a
 * mini block.
 *
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 * @pre The block pointer must not be NULL
 * @pre The size must be greater than 0 and should adhere to alignment
 * requirements
 * @post The header of the block is written with the specified size and alloc
 * status
 * @post does not alter the location of the block pointer or the contents of the
 * payload space
 */
static void write_block(block_t *block, size_t size, bool alloc,
                        bool prev_alloc, bool mini, bool prev_mini) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    block->header = pack(size, alloc, prev_alloc, mini, prev_mini);

    // Set the footer only if the block is free and not a mini block
    if (!alloc && !mini) {
        word_t *footerp = header_to_footer(block);

        // Ensure that the footer pointer is within the heap bounds before
        // dereferencing
        if ((char *)footerp >= (char *)mem_heap_lo() &&
            (char *)footerp < (char *)mem_heap_hi()) {
            *footerp = pack(size, alloc, prev_alloc, mini, prev_mini);
        }
    }

    block_t *next_block = find_next(block);
    update_next_block_header(next_block, alloc, mini);
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * Determine if a block is free.
 * @param block The block to check.
 * @return True if the block is free, false otherwise.
 */
static bool is_block_free(block_t *block) {
    return !get_alloc(block);
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * locate the block immediately preceding a given block in the heap's memory
 * layout.
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 * @pre The block is not the prologue
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_prev on the first block in the heap");
    if (get_prev_mini(block)) {
        block_t *prev = (block_t *)((char *)block - min_block_size);
        return prev;
    }
    word_t *footerp = find_prev_footer(block);
    return footer_to_header(footerp);
}

/**
 * @brief
 * Determines which segregated free list is apppriate for a block of the given
 * size
 * @param[in] block pointer to the block that is to be added to the free list.
 * @return an integer index representing the segregated free list that is best
 * suited for blocks of the given size.
 */
static int get_fl_index(size_t size) {
    if (size <= 0x20)
        return 0;
    else if (size <= 0x40)
        return 1;
    else if (size <= 0x80)
        return 2;
    else if (size <= 0x100)
        return 3;
    else if (size <= 0x200)
        return 4;
    else if (size <= 0x400)
        return 5;
    else if (size <= 0x800)
        return 6;
    else if (size <= 0x1000)
        return 7;
    else if (size <= 0x2000)
        return 8;
    else if (size <= 0x4000)
        return 9;
    else if (size <= 0x8000)
        return 10;
    else if (size <= 0x10000)
        return 11;
    else if (size <= 0x20000)
        return 12;
    else if (size <= 0x40000)
        return 13;
    else
        return 14;
}

/**
 * @brief add a free list
 * @param[in] block pointer to the block that is to be added to the free list.
 */
static void free_list_add(block_t *block) {
    int index = get_fl_index(get_size(block));

    if (index == 0) {
        block->next_free = seg_free_list[index];
        seg_free_list[index] = block;
    } else {
        block->prev_free = NULL;
        block->next_free = seg_free_list[index];
        if (seg_free_list[index] != NULL)
            seg_free_list[index]->prev_free = block;
        seg_free_list[index] = block;
    }
}

// remove free list for mini blocks
static void free_list_remove_m(block_t *block) {
    int index = get_fl_index(get_size(block));
    block_t *current = seg_free_list[index];
    block_t *prev = NULL;

    // Traverse the list until find the block or reach the end of the list.
    while (current != NULL && current != block) {
        prev = current;
        current = current->next_free;
    }

    // If the block was found, adjust the pointers to remove it from the list.
    if (current == block) {
        // If the block is at the beginning of the list.
        if (prev == NULL) {
            seg_free_list[index] = current->next_free;
        } else {
            // If the block is in the middle or at the end of the list.
            prev->next_free = current->next_free;
        }
    }
    block->next_free = NULL;
}

/**
 * @brief remove a free list
 *
 */
static void free_list_remove(block_t *block) {
    int index = get_fl_index(get_size(block));
    if (index == 0) {
        free_list_remove_m(block);
        return;
    }

    // If block is at the start of its list
    if (block->prev_free == NULL) {
        seg_free_list[index] = block->next_free;
    } else {
        // Link the previous block to the next, removing current block from the
        // list
        block->prev_free->next_free = block->next_free;
    }

    // If block is not at the end of its list
    if (block->next_free != NULL) {
        block->next_free->prev_free = block->prev_free;
    }

    // set the next and previous pointers of the removed block to null
    block->next_free = NULL;
    block->prev_free = NULL;
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

/**
 * @brief Coalesce the free block with any adjacent free blocks.
 *
 * This function merges the given block with adjacent free blocks to the left
 * and right in the heap to reduce fragmentation and maintain larger chunks
 * of contiguous free memory. It handles four cases of adjacent free blocks:
 * no adjacent free blocks, the previous block is free, the next block is free,
 * and both adjacent blocks are free.
 *
 * @param[in] block The current free block to be coalesced.
 * @return Returns the starting address of the coalesced free block.
 * @pre block must not be NULL.
 * @pre block must be a free block and must be within heap boundaries.
 * @post The returned block must be free.
 * @post The returned block's size must be greater than or equal to the original
 * block's size.
 * @post If adjacent blocks were free, the returned block must include their
 * sizes.
 * @post No two consecutive free blocks exist in the heap after coalescing.
 */
static block_t *coalesce_block(block_t *block) {

    block_t *next_block = find_next(block);

    bool prev_alloc = get_prev_alloc(block);
    bool next_alloc = next_block && get_alloc(next_block);
    bool is_mini = false;

    size_t size = get_size(block);

    if (prev_alloc && next_alloc) {

    } else if (prev_alloc && !next_alloc) {
        free_list_remove(next_block);
        size += get_size(next_block);
    } else if (!prev_alloc && next_alloc) {
        block_t *prev_block = find_prev(block);
        free_list_remove(prev_block);
        size += get_size(prev_block);
        block = prev_block;
    } else if (!prev_alloc && !next_alloc) {
        block_t *prev_block = find_prev(block);
        free_list_remove(prev_block);
        free_list_remove(next_block);
        size += (get_size(prev_block) + get_size(next_block));
        block = prev_block;
    }

    if (size <= min_block_size) {
        is_mini = true;
    } else {
        is_mini = false;
    }

    write_block(block, size, false, get_prev_alloc(block), is_mini,
                get_prev_mini(block));
    free_list_add(block);

    return block;
}

/**
 * @brief extend the heap by a specific size
 *
 * Requests additional memory from the system using mem_sbrk and initializes
 * the new free block and the new epilogue header.
 * Then, it coalesces this new free block with the previous free block if
 * possible.
 *
 * @param[in] size The size by which to extend the heap.
 * @return A pointer to the newly allocated block, or NULL if the memory cannot
 * be extended.
 * @pre size > 0
 * @pre size is aligned to double word size
 * @post heap is extended by at least the size
 * @post new free block is created and initialized with the size
 * @post returns a pointer to the new block or NULL if unable to extend the
 * heap.
 *
 */
static block_t *extend_heap(size_t size) {

    void *bp; // start of the newly allocated space in the heap

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk((intptr_t)size)) == (void *)-1) {
        return NULL;
    }

    /*
     * we write the new block's header one word before bp to properly precede
     the payload with its metadata. The original size request includes the
     entire block (header, payload, footer).
     */

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);
    write_block(block, size, false, get_prev_alloc(block),
                size <= min_block_size, get_prev_mini(block));

    // Create new epilogue header
    block_t *block_next = find_next(block);
    write_epilogue(block_next);

    return coalesce_block(block);
}

/**
 * @brief Splits a given allocated block into two, where the first part remains
 * allocated with the specified size (asize), and the second part becomes a new
 * free block if there's enough space.
 *
 * optimize memory usage by splitting a larger block into a portion that
 * meets the allocation request and another portion that can be returned to the
 * free list
 *
 * @param[in] block The block to be split
 * @param[in] asize The size for the first part of the block after splitting
 * @pre block must be initially allocated
 * @pre asize must be aligned and <= block's current size
 *
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));
    /* Can you write a precondition about the value of asize? */
    dbg_requires(asize > 0 && asize <= get_size(block));
    dbg_requires((asize & (dsize - 1)) == 0); // Ensure asize is aligned

    size_t block_size = get_size(block);

    if ((block_size - asize) >= min_block_size) {
        block_t *block_next;
        // Preserve the current block's previous allocation status
        bool prev_alloc = get_prev_alloc(block);
        write_block(block, asize, true, prev_alloc, get_mini(block),
                    get_prev_mini(block));

        block_next = find_next(block);
        bool is_mini = false;

        if ((block_size - asize) == min_block_size) {
            is_mini = true;
        }

        write_block(block_next, block_size - asize, false, true, is_mini,
                    asize <= min_block_size);
        free_list_add(block_next);
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief Searches the heap for a free block that fits the requested size
 *
 * This function looks for a free block of sufficient size to meet an allocation
 * request. It implements a best-fit search within a specified size threshold,
 * aiming to minimize waste while ensuring efficient allocation.
 *
 * @param[in] asize The size of the block needed
 * @return  A pointer to a suitable free block if found; otherwise, NULL.
 * @pre `asize` must be greater than 0 and aligned according to the allocator's
 * requirements.
 * @post The returned block, if any, remains marked as free in the segregated
 * list.
 *
 */
static block_t *find_fit(size_t asize) {
    int index = get_fl_index(asize);
    block_t *best_fit = NULL;
    size_t s_diff = SIZE_MAX;

    size_t threshold = asize / 32;

    // Start searching in the appropriate list based on the block size
    for (int i = index; i < NUM_FREE_LISTS && best_fit == NULL; i++) {
        block_t *current_list = seg_free_list[i];
        for (block_t *block = current_list; block != NULL;
             block = block->next_free) {
            size_t block_size = get_size(block);

            // Check if the block fits
            if (asize <= block_size) {
                size_t diff = block_size - asize;

                // If the difference is smaller than the current smallest,
                // update best_fit
                if (diff < s_diff) {
                    best_fit = block;
                    s_diff = diff;

                    // If the difference is within our threshold, we're done
                    // searching
                    if (diff <= threshold) {
                        return best_fit;
                    }
                }
            }
        }
    }

    // return best fit or null
    return best_fit;
}

/*print heap*/
static void print_heap() {

    dbg_printf("HEAP:\n");
    for (block_t *start = (block_t *)(((char *)mem_heap_lo() + 8));
         get_size(start) != 0; start = find_next(start)) {
        dbg_printf("address %lx size %lu is %s allocated", (unsigned long)start,
                   get_size(start), get_alloc(start) ? "" : "not");
        dbg_printf("prev_alloc: %s, prev_mini: %s\n",
                   get_prev_alloc(start) ? "true" : "false",
                   get_prev_mini(start) ? "true" : "false");
        if (!get_alloc(start)) {
            dbg_printf(" prev pointer %lx next pointer %lx \n",
                       (unsigned long)start->prev_free,
                       (unsigned long)start->next_free);
        } else {
            dbg_printf("\n");
        }
    }
}

/**
 * @brief checks the consistency of the heap
 *
 *  Verifies the heap data structure including the prologue and epilogue blocks,
 * correct alignment of each block, and that all blocks in the free list are
 * indeed marked as free. Also checks for coalescing
 *
 * @param[in] line line number from which this function was called
 * @return True if the heap is consistent, False otherwise
 * @pre Heap has been initialized
 * @post Function does not modify the heap or any of its blocks
 */
bool mm_checkheap(int line) {
    print_heap();

    dbg_requires(line >= 1);
    block_t *current = (block_t *)(((char *)mem_heap_lo() + 8) - wsize);

    // Check prologue block for size and allocation status
    if (!(get_size(current) == 0 && get_alloc(current))) {
        dbg_printf("Error: Invalid prologue at line %d\n", line);
        return false;
    }

    current = (block_t *)((char *)mem_heap_lo() + 8);
    // Traverse the entire heap
    while (get_size(current) > 0) {
        // Check block alignment
        if ((size_t)current % wsize != 0) {
            dbg_printf("Error: Block at %p is not aligned at line %d\n",
                       (void *)current, line);
            return false;
        }

        // check block is within heap boundaries
        if (current < (block_t *)mem_heap_lo() ||
            current > (block_t *)mem_heap_hi()) {
            dbg_printf("Error: Block at %p is out of heap bounds at line %d\n",
                       (void *)current, line);
            return false;
        }

        // For free blocks, check header and footer consistency
        if (!get_alloc(current)) {
            word_t *footer = header_to_footer(current);
            if (current->header != *footer) {
                dbg_printf("Error: Header/footer mismatch at %p at line %d\n",
                           (void *)current, line);
                return false;
            }
        }

        current = find_next(current);
    }

    // Traverse each segregated free list
    for (int i = 0; i < NUM_FREE_LISTS; i++) {
        block_t *slow = seg_free_list[i];
        block_t *fast = seg_free_list[i];

        // detect cycles
        while (slow != NULL && fast != NULL && fast->next_free != NULL) {
            slow = slow->next_free;            // Move slow pointer by one node
            fast = fast->next_free->next_free; // Move fast pointer by two nodes

            // Check for cycle
            if (slow == fast) {
                dbg_printf("Error: Detected cycle in free list %d at line %d\n",
                           i, line);
                return false;
            }
        }

        // Check if block is marked as free
        for (block_t *fblock = seg_free_list[i]; fblock != NULL;
             fblock = fblock->next_free) {
            if (get_alloc(fblock)) {
                dbg_printf(
                    "Error: Allocated block in free list at %p at line %d\n",
                    (void *)fblock, line);
                return false;
            }

            // Verify next/prev consistency
            if (fblock->next_free && fblock->next_free->prev_free != fblock) {
                dbg_printf("Error: Inconsistent next/prev pointers in free "
                           "list at %p at line %d\n",
                           (void *)fblock, line);
                return false;
            }

            // Verify blocks fall within the expected size range for the list
            size_t size = get_size(fblock);
            if (i < NUM_FREE_LISTS -
                        1) { // Exclude the last list since it's a catch-all
                if (get_fl_index(size) != i) {
                    dbg_printf("Error: Block at %p in list %d has size %lu, "
                               "out of expected range\n",
                               (void *)fblock, i, size);
                    return false;
                }
            }
        }
    }

    return true;
}

/**
 * @brief Initializes the memory manager
 *
 * * This function sets up the initial empty heap, including the prologue and
 * epilogue, and extends the heap by a predefined chunk size. It initializes
 * the segregated free list array to ensure each list starts as NULL, indicating
 * they are empty.
 *
 * @return True if the initialization is successful, false otherwise.
 */
bool mm_init(void) {
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, true, false, true); // Heap prologue (block footer)
    start[1] =
        pack(0, true, true, false, false); // Heap epilogue (block header)

    // Initialize each segregated free list to NULL
    for (int i = 0; i < NUM_FREE_LISTS; i++) {
        seg_free_list[i] = NULL;
    }

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }

    return true;
}

/**
 * @brief Allocates a block of memory of at least the specified size.
 *
 * Adjusts the requested size to meet alignment requirements
 * and searches for a suitable free block using a segregated fit strategy.
 * If no suitable block is found, it extends the heap.
 *
 * @param[in] size size The requested size of the payload.
 * @return pointer to the allocated block's payload if successful; otherwise,
 * NULL.
 */
void *malloc(size_t size) {
    dbg_printf("MALLOC: %lu\n", size);
    dbg_requires(mm_checkheap(__LINE__));

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if ((block_t *)((char *)mem_heap_lo() + 8) == NULL) {
        if (!(mm_init())) {
            dbg_printf("Problem initializing heap. Likely due to sbrk");
            return NULL;
        }
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust the block size to include overhead and meet alignment
    // requirements.
    asize = max(round_up(size + wsize, dsize), min_block_size);
    bool is_mini = asize <= min_block_size;

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    };
    free_list_remove(block);

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);
    write_block(block, block_size, true, prev_alloc, is_mini, prev_mini);

    // Try to split the block if too large
    split_block(block, asize);

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief Frees a block of memory, making it available for future allocations.
 *
 * marks a previously allocated block as free and attempts to
 * coalesce it with adjacent free blocks to reduce fragmentation. It updates
 * the block's header to reflect its new status and adjusts the free list
 * accordingly.
 *
 * @param[in] bp A pointer to the payload of the block to be freed.
 */
void free(void *bp) {
    dbg_printf("FREE: %lx\n", (unsigned long)payload_to_header(bp));
    dbg_requires(mm_checkheap(__LINE__));

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Get the allocation and mini status of the previous block
    bool prev_alloc = get_prev_alloc(block);
    bool prev_mini = get_prev_mini(block);

    // Mark the block as free
    write_block(block, size, false, prev_alloc, get_mini(block), prev_mini);

    // Try to coalesce the block with its neighbors
    block = coalesce_block(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief Reallocates a block of memory to a new size, preserving existing data
 * up to the minimum of the old and new sizes.
 *
 * This function adjusts the size of the allocated memory block pointed to by
 * `ptr` to `size` bytes. The contents of the block are preserved up to the
 * lesser of the new and old sizes. If `ptr` is NULL, the function behaves like
 * `malloc`. If `size` is 0, the function behaves like `free`.
 *
 * @param[in] ptr A pointer to the memory block to be reallocated. Can be NULL.
 * @param[in] size size The new size for the memory block in bytes.
 * @return A pointer to the newly allocated memory block, or NULL if allocation
 * fails.
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief Allocates memory for an array of elements of a certain size and
 * initializes all bytes to zero.
 *
 * This function allocates memory sufficient to hold an array of `elements`
 * elements, each of which is `size` bytes in size. The function then
 * initializes all bits in the allocated memory to 0.
 *
 * @param[in] elements The number of elements to allocate.
 * @param[in] size The size of each element.
 * @return A pointer to the allocated memory initialized to zero, or NULL if the
 * allocation fails or is unnecessary.
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
