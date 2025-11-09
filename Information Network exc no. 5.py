import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict


def recursive_series_mod(seed1, seed2, modulus=1404, num_terms=1000):
    """
    Generate recursive series: a_i = (a_{i-1} + a_{i-2}) % modulus
    Uses recursive generator function
    """

    def series_generator():
        a, b = seed1, seed2
        yield a
        yield b

        for _ in range(num_terms - 2):
            a, b = b, (a + b) % modulus
            yield b

    return list(series_generator())


def analyze_periodicity(series, modulus=1404):
    """
    Analyze periodicity of the recursive series
    """
    # Look for repeating patterns to determine period
    for period in range(1, len(series) // 2):
        is_periodic = True
        for i in range(period, len(series)):
            if series[i] != series[i % period]:
                is_periodic = False
                break
        if is_periodic:
            return period

    # If no exact period found, return approximate period using autocorrelation
    return find_approximate_period(series)


def find_approximate_period(series):
    """
    Find approximate period using autocorrelation
    """
    # Normalize series for autocorrelation
    normalized = np.array(series) / max(series)

    # Compute autocorrelation
    autocorr = np.correlate(normalized, normalized, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Find peaks (excluding zero lag)
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append(i)

    if len(peaks) > 1:
        return peaks[1]  # First non-zero peak
    return len(series)  # No clear period found


def generate_frequency_analysis(seeds_list, modulus=1404, num_terms=1000):
    """
    Generate comprehensive frequency analysis for multiple seed pairs
    """
    results = {}

    for i, (seed1, seed2) in enumerate(seeds_list):
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS {i + 1}: Seeds ({seed1}, {seed2})")
        print(f"{'=' * 60}")

        # Generate series
        series = recursive_series_mod(seed1, seed2, modulus, num_terms)

        # Calculate frequency distribution
        frequency = Counter(series)

        # Calculate periodicity
        period = analyze_periodicity(series)

        # Store results
        results[(seed1, seed2)] = {
            'series': series,
            'frequency': frequency,
            'period': period,
            'unique_values': len(frequency),
            'max_frequency': max(frequency.values()) if frequency else 0,
            'min_frequency': min(frequency.values()) if frequency else 0
        }

        # Print analysis
        print(f"Series length: {len(series)}")
        print(f"Periodicity: {period}")
        print(f"Unique values: {len(frequency)}")
        print(f"Value range: {min(series)} - {max(series)}")
        print(f"Max frequency: {max(frequency.values())}")
        print(f"Min frequency: {min(frequency.values())}")

        # Show most common values
        print("\nTop 10 most frequent values:")
        for value, freq in frequency.most_common(10):
            print(f"  {value}: {freq} times ({freq / num_terms * 100:.2f}%)")

    return results


def plot_frequency_distribution(results, seeds_list):
    """
    Plot frequency distributions for all seed pairs
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (seed1, seed2) in enumerate(seeds_list):
        if i >= len(axes):
            break

        data = results[(seed1, seed2)]
        frequency = data['frequency']

        # Create sorted frequency plot
        values = sorted(frequency.keys())
        frequencies = [frequency[v] for v in values]

        axes[i].bar(values, frequencies, alpha=0.7, color='skyblue')
        axes[i].set_title(f'Seeds ({seed1}, {seed2}) - Period: {data["period"]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_series_patterns(results, seeds_list, num_points=200):
    """
    Plot the first few terms of each series to visualize patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (seed1, seed2) in enumerate(seeds_list):
        if i >= len(axes):
            break

        series = results[(seed1, seed2)]['series']

        axes[i].plot(series[:num_points], 'o-', markersize=3, linewidth=1)
        axes[i].set_title(f'Series Pattern - Seeds ({seed1}, {seed2})')
        axes[i].set_xlabel('Term Index')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def advanced_period_analysis(seed1, seed2, modulus=1404, max_terms=5000):
    """
    Advanced period detection using state tracking
    """
    print(f"\n{'=' * 60}")
    print(f"ADVANCED PERIOD ANALYSIS: Seeds ({seed1}, {seed2})")
    print(f"{'=' * 60}")

    seen_states = {}
    series = []
    a, b = seed1, seed2

    series.append(a)
    series.append(b)
    seen_states[(a, b)] = 0

    period_found = False

    for i in range(2, max_terms):
        a, b = b, (a + b) % modulus
        series.append(b)

        current_state = (a, b)

        if current_state in seen_states:
            start_period = seen_states[current_state]
            period = i - start_period
            print(f"✓ Period found: {period}")
            print(f"  Starts at term: {start_period}")
            print(f"  Repeats at term: {i}")
            print(f"  Full period sequence: {series[start_period:start_period + period]}")
            period_found = True
            break

        seen_states[current_state] = i

    if not period_found:
        print(f"✗ No period found within {max_terms} terms")
        period = max_terms

    return series, period


# Main execution
if __name__ == "__main__":
    # Different seed pairs for analysis
    seeds_list = [
        (1, 1),  # Classic Fibonacci mod
        (0, 1),  # Standard Fibonacci starting point
        (13, 21),  # Fibonacci-like
        (7, 19),  # Prime seeds
    ]

    print("RECURSIVE SERIES FREQUENCY ANALYSIS")
    print("Series: a_i = (a_{i-1} + a_{i-2}) % 1404")
    print("=" * 60)

    # Generate frequency analysis for all seed pairs
    results = generate_frequency_analysis(seeds_list, modulus=1404, num_terms=1000)

    # Plot frequency distributions
    plot_frequency_distribution(results, seeds_list)

    # Plot series patterns
    plot_series_patterns(results, seeds_list)

    # Advanced period analysis for first seed pair
    advanced_series, exact_period = advanced_period_analysis(1, 1, modulus=1404)

    # Additional statistical analysis
    print(f"\n{'=' * 60}")
    print("STATISTICAL SUMMARY")
    print(f"{'=' * 60}")

    for (seed1, seed2), data in results.items():
        frequency = data['frequency']
        total_terms = sum(frequency.values())
        coverage = len(frequency) / 1404 * 100

        print(f"\nSeeds ({seed1}, {seed2}):")
        print(f"  Period: {data['period']}")
        print(f"  Coverage: {coverage:.2f}% of possible values")
        print(f"  Most frequent: {frequency.most_common(1)[0]}")
        print(f"  Least frequent: {frequency.most_common()[-1]}")

        # Calculate entropy (measure of uniformity)
        probabilities = [f / total_terms for f in frequency.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        max_entropy = np.log2(len(frequency))
        print(f"  Entropy: {entropy:.3f} bits (max possible: {max_entropy:.3f})")

    # Generate 1000 periodicity samples
    print(f"\n{'=' * 60}")
    print("GENERATING 1000 PERIODICITY SAMPLES")
    print(f"{'=' * 60}")

    period_samples = []
    for i in range(1000):
        seed1 = i % 100 + 1  # Vary seeds systematically
        seed2 = (i * 7) % 100 + 1
        series = recursive_series_mod(seed1, seed2, modulus=1404, num_terms=1000)
        period = analyze_periodicity(series)
        period_samples.append(period)

        if i < 10:  # Show first 10 samples
            print(f"Sample {i + 1}: seeds ({seed1}, {seed2}) -> period: {period}")

    print(f"\nPeriodicity Statistics over 1000 samples:")
    print(f"  Average period: {np.mean(period_samples):.2f}")
    print(f"  Standard deviation: {np.std(period_samples):.2f}")
    print(f"  Min period: {min(period_samples)}")
    print(f"  Max period: {max(period_samples)}")

    # Plot periodicity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(period_samples, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Periodicity over 1000 Samples')
    plt.xlabel('Period')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()