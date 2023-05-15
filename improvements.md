# Ideas for improvements

- Generate an index on the fly
    - The index stores chunks with the min/max values of each attribute in that chunk
    - Chunks could be e.g. 10k points in size
    - While scanning first (without an index) we can create a chunk header for each chunk, but only for the queried attribute(s)
        - So if we query by bounds, we compute an AABB for each chunk. If we query by object class, we create a class histogram, and so on...
    - Upon further scans, we first consult the index to find the matching chunks
    - Within each chunk, we can potentially refine the chunk, e.g. by calculating first half and second half chunks, and compare them if they are sufficiently different
    - For uncompressed data, we could also restructure the data (i.e. sort the points), but this only works for a single attribute
- The search should work stream-based so that we can stream the results into whatever application we want
    - In particular relevant for rendering / sending the data to a remote application

## TODOs

- Refinement does not work currently. I started something but then got stuck in too many optimizations. Would have to go back to the concept phase
- IF I really want to implement this, maybe start with a really small, unoptimized POC and go from there? e.g. only index positions and only down to LAZ-block granularity. Demonstrating a progressive index for LAZ blocks itself could already be a good result, as this could counteract the problems that the regular ad-hoc query implementation has with compressed files

## Session 14.05.23

- Habe ein ganz allgemeines Index-Refinement für WITHIN Position3D implementiert, die Refinements werden erstellt, allerdings noch nicht in den Index übernommen (da fehlt nur noch eine Implementierung von `BlockIndex::apply_refinements`)
- Habe runtime logging implementiert, um zu gucken wie sich die Laufzeit für die Query-Evaluierung, Index refinement, und Datenextraktion unterscheiden
    - Datenextraktion ist Faktor 5-10 intensiver als Query Evaluierung, und Faktor 3-5 intensiver als Refinement. Refinement ca. 2x so intensiv wie Evaluation
- Habe dann schon angefangen darüber nachzudenken, wie man Extraktion effizienter machen kann. Profiler sagt, dass es überwiegend `memmove` instructions sind, macht auch Sinn, ist eigentlich nur Daten von Buffer zu Buffer kopieren. Ideen:
    - Statt in einen `InterleavedPointBuffer` zu gehen könnte man direkt LAS point records extrahieren
    - Der Extractor könnte einfach `Vec<Range<usize>>` zurückliefern, und die Zielanwendung könnte dann die Konvertierung übernehmen. Könnte effizienter sein wenn ich z.B. direkt 3D Tiles extrahieren will oder sowas, dann muss ich nicht den Umweg über den pasture Buffer machen
- Ganz interessante Ergebnisse erstmal, TODOs für nächste Session wären:
    - Index refinement einbauen
    - Testszenarien aufbauen, die Index Refinement demonstrieren, also z.B. gleiche Query mehrmals, oder unterschiedliche Queries die aber evtl auch davon profitieren
        - Cool wäre, eine realistische Query zu haben aus einer 3D Anwendung raus, also z.B. Fibre3D
    - Index persistieren auf der Platte
    - Wiss. Frage: Wenn jetzt schon klar ist, dass data extraction den größten Teil der Zeit in Anspruch nimmt, in wieweit ergibt es dann noch Sinn, an Index/Query rumzubasteln? Sollte man nicht erstmal die Extraktion beschleunigen? 
    - Queries für LAZ ebenfalls implementieren, hier könnten die Ergebnisse anders aussehen und da könnte der Index deutlich wichtiger sein

## 15.05.23

- Vermutung: Ich muss erstmal eine MENGE Test-Szenarien aufbauen, sonst kriege ich die Komplexität von dem tool nicht in den Griff. Performance ist da zweitrangig, wirklich erstmal auflisten und implementieren, was ich alles gerne demonstrieren möchte
    - AABB queries haben wir ja schon, die sind einigermaßen trivial
    - LOD queries
        - Wie sieht die zugehörige QueryExpression aus? LOD matcht vermutlich so gut wie alle Blocks, da hilft der Index wenig. Man bräuchte also eher einen 'Importance-Index' (z.B. mit reverse-morton sorted Dateien)
    - Classification queries
    - Kombinierte Queries
    - Anwendungsfälle??
        - Direkte Visualisierung wäre toll, eventuell mit Tobias' Renderer oder direkt Potree/Cesium?
- Mal mit Michel und Tobias reden über deren progressive indexing ideen, vllt kann man hier noch was kombinieren? Geht aber letztlich ja nur drum für meine Diss noch paar Resultate zu kriegen
- Mit Kevin nochmal diskutieren was man aufbauend auf seinem Cloud-based tool noch machen könnte
    - Könnte dieses Tool hier auch was für die Cloud sein? 