    def generate_response(self, q, ctx, prompt_instr=""):
        # Note: Citation logic is handled separately by dedicated citation functions
        p = f"""{prompt_instr}

        You are a precise tender document analysis assistant. Your task is to answer questions based ONLY on the provided context from tender documents.

        CONTEXT:
        {ctx}

        QUESTION: {q}

        INSTRUCTIONS:
        1. Answer the question clearly and concisely using ONLY information from the context above
        2. If the context doesn't contain enough information, explicitly state: "The provided context does not contain sufficient information to answer this question"
        3. Do NOT make assumptions or add information not present in the context
        4. Reference specific sources when possible (e.g., "According to [Source/Page]...")
        5. If there are multiple relevant points, organize them with bullet points or numbered lists
        6. For numerical data, dates, or specific terms, quote them exactly as they appear in the context
        7. Maintain a professional and objective tone

        ANSWER:"""
        return self.llm.invoke([HumanMessage(content=p)]).content