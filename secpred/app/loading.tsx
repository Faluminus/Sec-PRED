export default function Loading() {
    // You can add any UI inside Loading, including a Skeleton.
    return (
        <div className="flex justify-center items-center w-screen h-screen">
            <div className="dna-spinner loader">
                <div className="wrapper">
                    <div className="row row-1">
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    </div>
                    <div className="row row-2">
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    <span className="span"></span>
                    </div>
                </div>
            </div>
        </div>
    )
  }