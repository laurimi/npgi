// combinations.hpp

// MIT License

// Copyright (c) 2019 Mikko Lauri
// Copyright (c) 2018 Artem Amirkhanov

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef COMBINATIONS_HPP
#define COMBINATIONS_HPP

#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace pgi
{

// Helper to get concrete objects pointed to by combination
template <typename SetIterator>
std::vector<typename std::iterator_traits<SetIterator>::value_type> get_combination(const std::vector<SetIterator>& comb) {
  std::vector<typename std::iterator_traits<SetIterator>::value_type> c;
  for (const auto& elemIt : comb) c.push_back(*elemIt);
  return c;
}

// Precondition: all sets between iterators should be non-empty:
// contain at least 1 element
template <typename SetsIter>
class combinations
{
   public:
    typedef combinations<SetsIter> Combinations_type;
    typedef typename std::iterator_traits<SetsIter>::value_type Set;
    typedef typename std::vector<typename Set::const_iterator> Combination;

    typedef Combination value_type;
    typedef long long difference_type;
    typedef size_t size_type;

    class const_iterator
    {
       public:
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef typename Combinations_type::difference_type difference_type;
        typedef typename Combinations_type::value_type value_type;
        typedef const Combination reference;
        typedef const Combination pointer;

        const_iterator() {}
        const_iterator(const const_iterator& other)
            : first_(other.first_), last_(other.last_), combination_(other.combination_)
        {
        }
        const_iterator(SetsIter first, SetsIter last, Combination combination)
            : first_(first), last_(last), combination_(combination)
        {
        }
        static const_iterator make_begin(const SetsIter first, const SetsIter last)
        {
            Combination combination;
            for (SetsIter it = first; it != last; ++it)
                combination.push_back(it->begin());
            return const_iterator(first, last, combination);
        }
        static const_iterator make_end(const SetsIter first, const SetsIter last)
        {
            SetsIter it = first;
            Combination combination(1, it->end());
            for (++it; it != last; ++it)
                combination.push_back(--it->end());
            return const_iterator(first, last, combination);
        }
        ~const_iterator() {}
        const_iterator& operator=(const const_iterator other)
        {
            swap(*this, other);
            return *this;
        }
        bool operator==(const const_iterator& other) const
        {
            return first_ == other.first_ && last_ == other.last_ && combination_ == other.combination_;
        }
        bool operator!=(const const_iterator& other) const { return !(*this == other); }
        const_iterator& operator++()
        {
            typename Combination::iterator combIt = combination_.begin();
            for (SetsIter it = first_; it != last_; ++it, ++combIt)
            {
                if (++(*combIt) != it->end())
                    return *this;
                *combIt = it->begin();
            }
            set_to_end_();
            return *this;
        }
        const_iterator& operator--()
        {
            typename Combination::iterator combIt = combination_.begin();
            for (SetsIter it = first_; it != last_; ++it, ++combIt)
            {
                if (*combIt != it->begin())
                {
                    --(*combIt);
                    return *this;
                }
                *combIt = --it->end();
            }
            set_to_begin_();
            return *this;
        }
        Combination operator*() const { return combination_; }
        Combination operator->() const { return combination_; }
        friend void swap(const_iterator& first, const_iterator& second)  // nothrow
        {
            std::swap(first.first_, second.first_);
            std::swap(first.last_, second.last_);
            std::swap(first.combination_, second.combination_);
        }
        void swap(Combinations_type other) { swap(*this, other); }
       private:
        void set_to_end_()
        {
            typename Combination::iterator combIt = combination_.begin();
            SetsIter it = first_;
            *(combIt++) = (it++)->end();
            for (; it != last_; ++it, ++combIt)
                *combIt = --it->end();
        }
        void set_to_begin_()
        {
            typename Combination::iterator combIt = combination_.begin();
            for (SetsIter it = first_; it != last_; ++it, ++combIt)
                *combIt = it->begin();
        }
        friend class combinations<SetsIter>;

       private:
        SetsIter first_;
        SetsIter last_;
        Combination combination_;
    };

    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    combinations() {}
    combinations(SetsIter first, SetsIter last) : first_(first), last_(last) {}
    combinations(const combinations<SetsIter>& other) : first_(other.first_), last_(other.last_) {}
    ~combinations() {}
    combinations& operator=(const combinations<SetsIter>& other)
    {
        swap(*this, other);
        return *this;
    }
    bool operator==(const combinations<SetsIter>& other) const
    {
        return first_ == other.first_ && last_ == other.last_;
    }
    bool operator!=(const combinations<SetsIter>& other) const { return !(*this == other); }
    const_iterator cbegin() const { return const_iterator::make_begin(first_, last_); }
    const_iterator begin() const { return cbegin(); }
    const_iterator cend() const { return const_iterator::make_end(first_, last_); }
    const_iterator end() const { return cend(); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return crbegin(); }
    const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return crend(); }
    friend void swap(combinations<SetsIter>& first, combinations<SetsIter>& second)  // nothrow
    {
        std::swap(first.first_, second.first_);
        std::swap(first.last_, second.last_);
    }
    void swap(combinations<SetsIter> other) { swap(*this, other); }
    size_type size() const { return std::accumulate(first_, last_, 1, mult_by_set_size_); }
    size_type max_size() const { return std::numeric_limits<size_type>::max(); }
    bool empty() const { return first_ == last_; }
   private:
    static size_type mult_by_set_size_(const size_type prev_sz, const Set& set) { return prev_sz * set.size(); }
   private:
    SetsIter first_;
    SetsIter last_;
};

template <typename Sets>
combinations<typename Sets::const_iterator> make_combinations(const Sets& data)
{
    return combinations<typename Sets::const_iterator>(data.cbegin(), data.cend());
}

}  // namespace pgi

#endif  // COMBINATIONS_HPP
