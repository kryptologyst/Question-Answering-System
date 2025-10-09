"""
Mock Database for Question Answering System
Contains diverse documents across various topics for testing QA capabilities
"""

import json
from typing import List, Dict, Any
from pathlib import Path

class MockDatabase:
    """Mock database containing diverse documents for QA testing"""
    
    def __init__(self):
        self.documents = self._load_sample_documents()
    
    def _load_sample_documents(self) -> List[Dict[str, Any]]:
        """Load sample documents covering various topics"""
        return [
            {
                "id": "ai_basics",
                "title": "Artificial Intelligence Fundamentals",
                "content": """
                Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
                that are programmed to think and learn like humans. The term may also be applied to any 
                machine that exhibits traits associated with a human mind such as learning and problem-solving. 
                AI can be categorized into three types: narrow AI, general AI, and super AI. Narrow AI is 
                designed to perform specific tasks, while general AI can understand, learn, and apply knowledge 
                across different domains. Super AI would surpass human intelligence in all areas.
                """,
                "category": "technology",
                "tags": ["AI", "machine learning", "intelligence", "automation"]
            },
            {
                "id": "machine_learning",
                "title": "Machine Learning Overview",
                "content": """
                Machine Learning (ML) is a subset of artificial intelligence that focuses on algorithms 
                and statistical models that enable computer systems to improve their performance on a specific 
                task through experience. There are three main types of machine learning: supervised learning, 
                unsupervised learning, and reinforcement learning. Supervised learning uses labeled training 
                data, unsupervised learning finds hidden patterns in data without labels, and reinforcement 
                learning learns through interaction with an environment using rewards and penalties.
                """,
                "category": "technology",
                "tags": ["machine learning", "algorithms", "data science", "statistics"]
            },
            {
                "id": "climate_change",
                "title": "Climate Change and Global Warming",
                "content": """
                Climate change refers to long-term shifts in global temperatures and weather patterns. 
                While climate variations occur naturally, since the 1800s human activities have been the 
                main driver of climate change, primarily due to burning fossil fuels which generates 
                greenhouse gas emissions. The main greenhouse gases are carbon dioxide, methane, and nitrous 
                oxide. These gases trap heat in the atmosphere, causing global warming. Effects include 
                rising sea levels, extreme weather events, and ecosystem disruption.
                """,
                "category": "environment",
                "tags": ["climate", "global warming", "greenhouse gases", "environment"]
            },
            {
                "id": "space_exploration",
                "title": "Space Exploration History",
                "content": """
                Space exploration is the investigation of outer space by means of manned and unmanned 
                spacecraft. The space age began with the launch of Sputnik 1 by the Soviet Union in 1957. 
                Major milestones include Yuri Gagarin becoming the first human in space in 1961, the Apollo 
                11 moon landing in 1969, and the launch of the International Space Station in 1998. Modern 
                space exploration focuses on Mars missions, asteroid mining, and the search for extraterrestrial 
                life. Private companies like SpaceX are revolutionizing space travel with reusable rockets.
                """,
                "category": "science",
                "tags": ["space", "exploration", "NASA", "rockets", "Mars"]
            },
            {
                "id": "quantum_computing",
                "title": "Quantum Computing Fundamentals",
                "content": """
                Quantum computing is a type of computation that harnesses the collective properties of 
                quantum states, such as superposition, interference, and entanglement, to perform calculations. 
                Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or 
                qubits that can exist in multiple states simultaneously. This allows quantum computers to 
                process vast amounts of data and solve certain problems exponentially faster than classical 
                computers. Applications include cryptography, drug discovery, financial modeling, and optimization.
                """,
                "category": "technology",
                "tags": ["quantum", "computing", "qubits", "superposition", "entanglement"]
            },
            {
                "id": "renewable_energy",
                "title": "Renewable Energy Sources",
                "content": """
                Renewable energy comes from natural sources that are constantly replenished, such as 
                sunlight, wind, rain, tides, waves, and geothermal heat. Unlike fossil fuels, renewable 
                energy sources produce little to no greenhouse gas emissions. Solar energy harnesses 
                sunlight using photovoltaic cells or solar thermal systems. Wind energy uses turbines to 
                convert wind into electricity. Hydropower generates electricity from flowing water. 
                These technologies are becoming increasingly cost-effective and are crucial for reducing 
                carbon emissions and combating climate change.
                """,
                "category": "environment",
                "tags": ["renewable energy", "solar", "wind", "hydropower", "sustainability"]
            },
            {
                "id": "neuroscience",
                "title": "Neuroscience and Brain Research",
                "content": """
                Neuroscience is the scientific study of the nervous system, including the brain, spinal 
                cord, and networks of sensory nerve cells. The human brain contains approximately 86 billion 
                neurons connected by trillions of synapses. Key areas of research include understanding 
                consciousness, memory formation, neural plasticity, and brain-computer interfaces. Recent 
                advances in neuroimaging techniques like fMRI and EEG have revolutionized our understanding 
                of brain function. Applications include treating neurological disorders, developing AI systems, 
                and creating brain-machine interfaces for medical purposes.
                """,
                "category": "science",
                "tags": ["neuroscience", "brain", "neurons", "consciousness", "neuroimaging"]
            },
            {
                "id": "blockchain",
                "title": "Blockchain Technology",
                "content": """
                Blockchain is a distributed ledger technology that maintains a continuously growing list 
                of records, called blocks, which are linked and secured using cryptography. Each block 
                contains a cryptographic hash of the previous block, a timestamp, and transaction data. 
                This creates an immutable chain of data. Blockchain is the technology behind cryptocurrencies 
                like Bitcoin, but its applications extend far beyond digital currencies. It can be used 
                for supply chain management, voting systems, identity verification, and smart contracts.
                """,
                "category": "technology",
                "tags": ["blockchain", "cryptocurrency", "Bitcoin", "distributed ledger", "smart contracts"]
            },
            {
                "id": "genetics",
                "title": "Genetics and DNA",
                "content": """
                Genetics is the study of genes, genetic variation, and heredity in living organisms. 
                DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for the 
                development, functioning, growth, and reproduction of all known organisms. The human genome 
                contains approximately 3 billion base pairs and about 20,000-25,000 genes. Recent advances 
                in gene editing technologies like CRISPR-Cas9 have revolutionized genetic research and 
                opened possibilities for treating genetic diseases. Applications include personalized medicine, 
                genetic testing, and gene therapy.
                """,
                "category": "science",
                "tags": ["genetics", "DNA", "genes", "CRISPR", "genome", "heredity"]
            },
            {
                "id": "sustainable_agriculture",
                "title": "Sustainable Agriculture Practices",
                "content": """
                Sustainable agriculture is farming in sustainable ways that meet society's present food 
                and textile needs without compromising the ability of future generations to meet their own 
                needs. Practices include crop rotation, organic farming, integrated pest management, and 
                precision agriculture using GPS and sensors. Sustainable agriculture aims to maintain 
                soil health, conserve water, reduce chemical inputs, and promote biodiversity. It addresses 
                challenges like climate change, soil degradation, and food security while supporting 
                rural communities and maintaining ecosystem services.
                """,
                "category": "environment",
                "tags": ["sustainable agriculture", "organic farming", "soil health", "biodiversity", "food security"]
            }
        ]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the database"""
        return self.documents
    
    def get_documents_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get documents filtered by category"""
        return [doc for doc in self.documents if doc["category"] == category]
    
    def get_documents_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get documents containing a specific tag"""
        return [doc for doc in self.documents if tag.lower() in [t.lower() for t in doc["tags"]]]
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by title or content"""
        query_lower = query.lower()
        results = []
        for doc in self.documents:
            if (query_lower in doc["title"].lower() or 
                query_lower in doc["content"].lower() or
                any(query_lower in tag.lower() for tag in doc["tags"])):
                results.append(doc)
        return results
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_content_texts(self) -> List[str]:
        """Get all document contents as plain text for QA system"""
        return [doc["content"].strip() for doc in self.documents]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        categories = {}
        all_tags = []
        
        for doc in self.documents:
            category = doc["category"]
            categories[category] = categories.get(category, 0) + 1
            all_tags.extend(doc["tags"])
        
        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "unique_tags": len(set(all_tags)),
            "total_tags": len(all_tags)
        }
    
    def save_to_file(self, filepath: str):
        """Save the database to a JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load database from a JSON file"""
        instance = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            instance.documents = json.load(f)
        return instance

def main():
    """Test the mock database"""
    db = MockDatabase()
    
    print("📚 Mock Database Statistics:")
    print("=" * 40)
    stats = db.get_statistics()
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Categories: {stats['categories']}")
    print(f"Unique Tags: {stats['unique_tags']}")
    
    print("\n🔍 Sample Search Results:")
    print("-" * 30)
    search_results = db.search_documents("AI")
    for doc in search_results[:3]:
        print(f"📄 {doc['title']} ({doc['category']})")
        print(f"   Tags: {', '.join(doc['tags'])}")
        print()
    
    print("💾 Saving database to file...")
    db.save_to_file("data/documents.json")
    print("✅ Database saved successfully!")

if __name__ == "__main__":
    main()
