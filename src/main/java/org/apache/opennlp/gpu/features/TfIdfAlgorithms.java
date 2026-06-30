package org.apache.opennlp.gpu.features;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Shared TF-IDF algorithms used by multiple compute backends.
 *
 * <p>This helper intentionally computes a single scalar score per document
 * because {@code FeatureExtractionOperation#computeTfIdf(String[])} returns
 * {@code float[]} with one element per input document.</p>
 */
public final class TfIdfAlgorithms {

    private TfIdfAlgorithms() {
        // utility class
    }

    /**
     * Compute a TF-IDF score per document using smoothed IDF:
     * idf(t) = ln((N + 1) / (df(t) + 1)) + 1
     *
     * <p>The per-document scalar is the L2 norm of the document TF-IDF vector.</p>
     *
     * @param documents input documents
     * @return one TF-IDF score per document
     */
    public static float[] computeDocumentScores(String[] documents) {
        if (documents == null || documents.length == 0) {
            return new float[0];
        }

        int totalDocs = documents.length;
        List<String[]> tokenizedDocs = new ArrayList<>(totalDocs);
        Map<String, Integer> documentFrequency = new HashMap<>();

        // Pass 1: tokenize and build document frequency map.
        for (String doc : documents) {
            String[] tokens = tokenize(doc);
            tokenizedDocs.add(tokens);

            Set<String> seenInDoc = new HashSet<>();
            for (String token : tokens) {
                if (seenInDoc.add(token)) {
                    documentFrequency.merge(token, 1, Integer::sum);
                }
            }
        }

        // Pass 2: compute per-document TF-IDF L2 norm.
        float[] scores = new float[totalDocs];
        for (int docIndex = 0; docIndex < totalDocs; docIndex++) {
            String[] tokens = tokenizedDocs.get(docIndex);
            if (tokens.length == 0) {
                scores[docIndex] = 0.0f;
                continue;
            }

            Map<String, Integer> termFrequency = new HashMap<>();
            for (String token : tokens) {
                termFrequency.merge(token, 1, Integer::sum);
            }

            double sumSquares = 0.0;
            for (Map.Entry<String, Integer> termEntry : termFrequency.entrySet()) {
                String term = termEntry.getKey();
                int tfCount = termEntry.getValue();
                int df = documentFrequency.getOrDefault(term, 0);

                double tf = (double) tfCount / tokens.length;
                double idf = Math.log((totalDocs + 1.0) / (df + 1.0)) + 1.0;
                double tfidf = tf * idf;

                sumSquares += tfidf * tfidf;
            }

            scores[docIndex] = (float) Math.sqrt(sumSquares);
        }

        return scores;
    }

    private static String[] tokenize(String document) {
        if (document == null) {
            return new String[0];
        }

        String trimmed = document.trim().toLowerCase();
        if (trimmed.isEmpty()) {
            return new String[0];
        }

        return trimmed.split("\\s+");
    }
}
