from src.core.vector_db import VectorDatabase
from langchain.schema import Document

vdb = VectorDatabase()

# Add some realistic legal document content for better testing
legal_docs = [
    Document(
        page_content="""EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on [Date] between [Company Name], a [State] corporation ("Company"), and [Employee Name] ("Employee").

1. POSITION AND DUTIES
Employee is hired as [Title] and shall perform duties including but not limited to:
- Managing day-to-day operations
- Supervising staff and ensuring compliance with company policies
- Reporting to the Board of Directors on quarterly basis

2. COMPENSATION
- Base salary: $[Amount] per year, payable bi-weekly
- Performance bonus: Up to 20% of base salary based on annual performance review
- Benefits: Health insurance, dental, vision, 401(k) with 4% company match

3. TERMINATION
Either party may terminate this agreement with 30 days written notice. Company may terminate immediately for cause including:
- Breach of confidentiality
- Criminal conviction
- Failure to perform duties after written warning

4. CONFIDENTIALITY
Employee agrees to maintain strict confidentiality of all proprietary information, trade secrets, and customer data both during and after employment.

5. NON-COMPETE
Employee agrees not to work for direct competitors within 50 miles for 12 months after termination.""",
        metadata={'source': 'employment_agreement.pdf', 'page': 1, 'doc_type': 'employment_contract'}
    ),
    Document(
        page_content="""NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is made between [Disclosing Party] and [Receiving Party].

PURPOSE: To protect confidential information shared during business discussions regarding potential partnership opportunities.

CONFIDENTIAL INFORMATION includes:
- Technical specifications and proprietary methodologies
- Financial information, pricing structures, and business plans
- Customer lists, supplier information, and market research
- Any information marked as "Confidential" or that reasonably should be considered confidential

OBLIGATIONS:
1. Receiving Party shall not disclose confidential information to third parties
2. Information shall only be used for evaluation purposes
3. Reasonable security measures must be implemented to protect information
4. All materials must be returned upon request or termination of discussions

TERM: This agreement remains in effect for 5 years from the date of signing.

REMEDIES: Breach may result in irreparable harm warranting injunctive relief and monetary damages.""",
        metadata={'source': 'nda_agreement.pdf', 'page': 1, 'doc_type': 'nda'}
    ),
    Document(
        page_content="""SERVICE AGREEMENT

This Service Agreement governs the provision of consulting services between [Service Provider] and [Client].

SCOPE OF SERVICES:
- Strategic business consulting and market analysis
- Implementation of operational improvements
- Staff training and development programs
- Quarterly progress reports and recommendations

PAYMENT TERMS:
- Professional fees: $200 per hour
- Monthly retainer: $5,000 due on the 1st of each month
- Expenses: Client reimburses for reasonable business expenses with prior approval
- Late fees: 1.5% per month on overdue amounts

INTELLECTUAL PROPERTY:
- Work product created specifically for Client belongs to Client
- Provider retains rights to general methodologies and prior knowledge
- No license granted to Provider's proprietary tools and frameworks

LIMITATION OF LIABILITY:
Provider's liability limited to amount of fees paid in preceding 12 months. No liability for indirect, consequential, or punitive damages.

TERMINATION:
Either party may terminate with 30 days notice. Client pays for work completed through termination date.""",
        metadata={'source': 'service_agreement.pdf', 'page': 1, 'doc_type': 'service_contract'}
    )
]

print('Adding realistic legal document content...')
vdb.add_documents(legal_docs)
print('Added legal documents for testing')
print('Updated stats:', vdb.get_stats())
