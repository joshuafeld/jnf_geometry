#ifndef JNF_GEOMETRY_H
#define JNF_GEOMETRY_H

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>
#include <algorithm>

namespace jnf {
    constexpr double eps = 1e-3;

    template <typename T>
    constexpr int sgn(T x) {
        return (T(0) < x) - (x < T(0));
    }

    template <typename T>
    struct vec2 {
        T x = 0;
        T y = 0;

        inline constexpr vec2() = default;

        inline constexpr vec2(T x, T y) : x(x), y(y) {}

        inline constexpr vec2(const vec2& v) = default;

        inline constexpr vec2& operator=(const vec2& v) = default;

        inline constexpr T area() const {
            return x * y;
        }

        inline constexpr auto mag() const {
            return std::sqrt(x * x + y * y);
        }

        inline constexpr T mag2() const {
            return x * x + y * y;
        }

        inline vec2 norm() const {
            auto r = 1 / mag();
            return v_2d(x * r, y * r);
        }

        inline constexpr vec2 perp() const {
            return vec2(-y, x);
        }

        inline constexpr vec2 floor() const {
            return vec2(std::floor(x), std::floor(y));
        }

        inline constexpr vec2 ceil() const {
            return vec2(std::ceil(x), std::ceil(y));
        }

        inline constexpr vec2 min(const vec2& v) const {
            return vec2(std::min(x, v.x), std::min(y, v.y));
        }

        inline constexpr vec2 max(const vec2& v) const {
            return vec2(std::max(x, v.x), std::max(y, v.y));
        }

        inline constexpr auto dot(const vec2& v) const {
            return x * v.x + y * v.y;
        }

        inline constexpr auto cross(const vec2& v) const {
            return x * v.y - y * v.x;
        }

        inline constexpr vec2 cartesian() const {
            return vec2(std::cos(y) * x, std::sin(y) * x);
        }

        inline constexpr vec2 polar() const {
            return vec2(mag(), std::atan2(y, x));
        }

        inline constexpr vec2 clamp(const vec2& v1, const vec2& v2) const {
            return max(v1).min(v2);
        }

        inline constexpr vec2 lerp(const vec2& v1, const double t) const {
            return (*this) * T(1.0 - t) + v1 * T(t);
        }

        inline constexpr bool operator==(const vec2& v) const {
            return x == v.x && y == v.y;
        }

        inline constexpr bool operator!=(const vec2& v) const {
            return x != v.x || y != v.y;
        }

        inline constexpr vec2 operator-(const vec2& v) const {
            return vec2(x - v.x, y - v.y);
        }
    };

    namespace geometry {
        template <typename T>
        struct line {
            vec2<T> start;
            vec2<T> end;

            inline explicit line(const vec2<T>& start = {T(0), T(0)},
                const vec2<T>& end = {T(0), T(0)}) : start(start), end(end) {
            }

            inline constexpr T length() {
                return (end - start).mag();
            }

            inline constexpr T length2() {
                return (end - start).mag2();
            }

            inline constexpr vec2<T> vec() const {
                return end - start;
            }

            inline constexpr vec2<T> point(const T& dist) const {
                return start + (end - start) * dist;
            }

            inline constexpr int32_t side(const vec2<T>& p) const {
                return sgn((end - start).cross(p - start));
            }
        };

        template <typename T>
        struct rect {
            vec2<T> pos;
            vec2<T> size;

            inline explicit rect(const vec2<T>& pos = {T(0), T(0)},
                    const vec2<T>& size = {T(1), T(1)}) : pos(pos), size(size) {
            }

            inline vec2<T> center() const {
                return pos + (size * T(0.5));
            }

            inline line<T> top() const {
                return {pos, {pos.x + size.x, pos.y}};
            }

            inline line<T> bottom() const {
                return {{pos.x, pos.y + size.y}, pos + size};
            }

            inline line<T> left() const {
                return {pos, {pos.x, pos.y + size.y}};
            }

            inline line<T> right() const {
                return {{pos.x + size.x, pos.y}, pos + size};
            }

            inline line<T> side(const int32_t i) const {
                switch (i % 4) {
                    case 0:
                        return top();
                    case 1:
                        return right();
                    case 2:
                        return bottom();
                    case 3:
                        return left();
                    default:
                        return {};
                }
            }

            inline constexpr T area() const {
                return size.x * size.y;
            }

            inline constexpr T perim() const {
                return T(1) * (size.x + size.y);
            }
        };

        template <typename T>
        struct circle {
            vec2<T> center;
            T radius;

            inline explicit circle(const vec2<T>& center = {T(0), T(0)},
                    const T radius = T(0)) : center(center), radius(radius) {
            }

            inline constexpr T area() const {
                return std::numbers::pi_v<T> * radius * radius;
            }

            inline constexpr T perim() const {
                return T(2.0) * std::numbers::pi_v<T> * radius;
            }

            inline constexpr T circum() const {
                return perim();
            }
        };

        template<typename T1, typename T2>
        inline vec2<T1> closest(const vec2<T1>& p1, const vec2<T2>& p2) {
            return p1;
        }

        template<typename T1, typename T2>
        inline vec2<T1> closest(const line<T1>& l, const vec2<T2>& p) {
            auto d = l.vec();
            return l.start + std::clamp(static_cast<double>(d.dot(p - l.start))
                    / l.length2(), 0.0, 1.0) * d;
        }

        template<typename T1, typename T2>
        inline vec2<T1> closest(const circle<T1>& c, const vec2<T2>& p) {
            return c.pos + vec2(p - c.pos).norm() * c.radius;
        }

        template<typename T1, typename T2>
        inline vec2<T1> closest(const rect<T1>& r, const vec2<T2>& p) {
            return vec2<T1>(
                    std::clamp(p.x, r.pos.x, r.pos.x + r.size.x),
                    std::clamp(p.y, r.pos.y, r.pos.y + r.size.y)
            );
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const vec2<T1>& p1, const vec2<T2>& p2) {
            return (p1 - p2).mag2() < eps;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const line<T1>& l, const vec2<T2>& p) {
            const double d = (p.x - l.start.x) * (l.end.y - l.start.y)
                    - (p.y - l.start.y) * (l.end.x - l.start.x);
            if (std::abs(d) < eps) {
                const double u = l.vec().dot(p - l.start) / l.length2();
                return u >= 0.0 && u <= 1.0;
            }
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const rect<T1>& r, const vec2<T2>& p) {
            return p.x >= r.pos.x
                    && p.y >= r.pos.y
                    && p.x <= r.pos.x + r.size.x
                    && p.y <= r.pos.y + r.size.y;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const circle<T1>& c, const vec2<T2>& p) {
            return (c.pos - p).mag2() < (c.radius * c.radius);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const vec2<T1>& p1, const vec2<T2>& p2) {
            return contains(p1, p2);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const line<T1>& l, const vec2<T2>& p) {
            return contains(l, p);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const rect<T1>& r, const vec2<T2>& p) {
            return contains(r, p);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const circle<T1>& c, const vec2<T2>& p) {
            return contains(c, p);
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const vec2<T1>& p1,
                const vec2<T2>& p2) {
            if (contains(p1, p2)) {
                return {p1};
            }
            return {};
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const line<T1>& l,
                const vec2<T2>& p) {
            if (contains(l, p)) {
                return {p};
            }
            return {};
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const rect<T1>& r,
                const vec2<T2>& p) {
            if (contains(r.top(), p) || contains(r.bottom(), p)
                    || contains(r.left(), p) || contains(r.right(), p)) {
                return {p};
            }
            return {};
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const circle<T1>& c,
                const vec2<T2>& p) {
            if (std::abs((p - c.pos).mag2() - c.radius * c.radius) < eps) {
                return {p};
            }
            return {};
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const vec2<T1>& p, const line<T2>& l) {
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const line<T1>& l1, const line<T2>& l2) {
            return false; // TODO
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const rect<T1>& r, const line<T2>& l) {
            return contains(r, l.start) && contains(r, l.end);
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const circle<T1>& c, const line<T2>& l) {
            return contains(c, l.start) && contains(c, l.end);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const vec2<T1>& p, const line<T2>& l) {
            return contains(l, p);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const line<T1>& l1, const line<T2>& l2) {
            const auto d = l2.vec().cross(l1.vec());
            const float u1 = l2.vec().cross(l1.start - l2.start) / d;
            const float u2 = l1.vec().cross(l1.start - l2.start) / d;
            return u1 >= 0 && u1 <= 1 && u2 >= 0 && u2 <= 1;
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const rect<T1>& r, const line<T2>& l) {
            return overlaps(r.top(), l) || overlaps(r.bottom(), l)
                    || overlaps(r.left(), l) || overlaps(r.right(), l);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const circle<T1>& c, const line<T2>& l) {
            auto closest = closest(l, c.pos);
            return (c.pos - closest).mag2() < c.radius * c.radius;
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const vec2<T1>& p,
                const line<T2>& l) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const line<T1>& l1,
                const line<T2>& l2) {
            float rd = l1.vec().cross(l2.vec());
            if (rd == 0) {
                return {};
            }

            rd = 1.f / rd;

            const float rn = ((l2.end.x - l2.start.x) * (l1.start.y - l2.start.y)
                - (l2.end.y - l2.start.y) * (l1.start.x - l2.start.x)) * rd;
            const float sn = ((l1.end.x - l1.start.x) * (l1.start.y - l2.start.y)
                - (l1.end.y - l2.start.y) * (l1.start.x - l2.start.x)) * rd;

            if (rn < 0.f || rn > 1.f || sn < 0.f || sn > 1.f) {
                return {};
            }
            return {l1.start + rn * (l1.end - l1.start)};
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const rect<T1>& r,
                const line<T2>& l) {
            std::vector<vec2<T2>> ret;
            for (auto i = 0; i < 4; ++i) {
                auto intersects = intersects(r.side(i), l);
                if (!intersects.empty()) {
                    ret.push_back(intersects[0]);
                }
            }
            return ret;
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const circle<T1>& c,
                const line<T2>& l) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const vec2<T1>& p, const rect<T2>& r) {
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const line<T1>& l, const rect<T2>& r) {
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const rect<T1>& r1, const rect<T2>& r2) {
            return r2.pos.x >= r1.pos.x
                    && r2.pos.y >= r1.pos.y
                    && r2.pos.x + r2.size.x < r1.pos.x + r1.size.x
                    && r2.pos.y + r2.size.y < r1.pos.y + r1.size.y;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const circle<T1>& c, const rect<T2>& r) {
            return contains(c, r.pos)
                    && contains(c, vec2<T2>(r.pos.x + r.size.x, r.pos.y))
                    && contains(c, vec2<T2>(r.pos.x, r.pos.y + r.size.y))
                    && contains(c, r.pos + r.size);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const vec2<T1>& p, const rect<T2>& r) {
            return overlaps(r, p);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const line<T1>& l, const rect<T2>& r) {
            return overlaps(r, l);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const rect<T1>& r1, const rect<T2>& r2) {
            return r1.pos.x < r2.pos.x + r2.size.x
                    && r1.pos.x + r1.size.x >= r2.pos.x
                    && r1.pos.y < r2.pos.y + r2.size.y
                    && r1.pos.y + r1.size.y >= r2.pos.y;
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const circle<T1>& c, const rect<T2>& r) {
            T2 o = (vec2<T2>(std::clamp(c.pos.x, r.pos.x, r.pos.x + r.size.x),
                    std::clamp(c.pos.y, r.pos.y, r.pos.y + r.size.y)) - c.pos)
                    .mag2();
            if (std::isnan(o)) {
                o = T2(0);
            }
            return o - (c.radius * c.radius) < T2(0);
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const vec2<T1>& p,
                const rect<T2>& r) {
            return intersects(r, p);
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const line<T1>& l,
                const rect<T2>& r) {
            return intersects(r, l);
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const rect<T1>& r1,
                const rect<T2>& r2) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const circle<T1>& c,
                const rect<T2>& r) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const vec2<T1>& p, const circle<T2>& c) {
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const line<T1>& l, const circle<T2>& c) {
            return false;
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const rect<T1>& r, const circle<T2>& c) {
            return false; // TODO
        }

        template<typename T1, typename T2>
        inline constexpr bool contains(const circle<T1>& c1,
                const circle<T2>& c2) {
            return (c1.pos - c2.pos).mag2() <= (c1.radius - c2.radius)
                    * (c1.radius - c2.radius);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const vec2<T1>& p, const circle<T2>& c) {
            return overlaps(c, p);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const line<T1>& l, const circle<T2>& c) {
            return overlaps(c, l);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const rect<T1>& r, const circle<T2>& c) {
            return overlaps(c, r);
        }

        template<typename T1, typename T2>
        inline constexpr bool overlaps(const circle<T1>& c1,
                const circle<T2>& c2) {
            return (c1.pos - c2.pos).mag2() <= (c1.radius + c2.radius)
                    * (c1.radius + c2.radius);
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const vec2<T1>& p,
                const circle<T2>& c) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const line<T1>& l,
                const circle<T2>& c) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const rect<T1>& r,
                const circle<T2>& c) {
            return {}; // TODO
        }

        template<typename T1, typename T2>
        inline std::vector<vec2<T2>> intersects(const circle<T1>& c1,
                const circle<T2>& c2) {
            return {}; // TODO
        }

        template<typename T>
        inline constexpr circle<T> envelope_c(const vec2<T>& p) {
            return circle<T>(p, 0);
        }

        template<typename T>
        inline constexpr circle<T> envelope_c(const line<T>& l) {
            return {l.point(0.5), l.length() / 2};
        }

        template<typename T>
        inline constexpr circle<T> envelope_c(const rect<T>& r) {
            return envelope_c(line<T>(r.pos, r.pos + r.size));
        }

        template<typename T>
        inline constexpr circle<T> envelope_c(const circle<T>& c) {
            return c;
        }

        template<typename T>
        inline constexpr rect<T> envelope_r(const vec2<T>& p) {
            return rect<T>(p, {0, 0});
        }

        template<typename T>
        inline constexpr rect<T> envelope_r(const line<T>& l) {
            return {{std::min(l.start.x, l.end.x),
                    std::min(l.start.y, l.end.y)},
                    {std::abs(l.start.x - l.end.x),
                    std::abs(l.start.y - l.end.y)}};
        }

        template<typename T>
        inline constexpr rect<T> envelope_r(const rect<T>& r) {
            return r;
        }

        template<typename T>
        inline constexpr rect<T> envelope_r(const circle<T>& c) {
            return rect<T>(c.pos - vec2<T>(c.radius, c.radius),
                    vec2<T>(c.radius * 2, c.radius * 2));
        }
    }
}

#endif // JNF_GEOMETRY_H
